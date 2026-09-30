#!/usr/bin/env python3
"""Frozen TRAIN screening and full confirmation for same-budget state repairs."""
from __future__ import annotations
import argparse
import contextlib
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime, resurface_data as data, resurface_native as native
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from mamba2_recall.state_quant import StateQuant
from mamba2_recall.state_quant_dense3 import DenseStateQuant
from prepare_quant_first import load_train_tokens, TRAIN_SHA, PROSE_MANIFEST_SHA
from evaluate_quant_first import load_candidate, FrozenBase, compare_pair, check_restoration, VALIDATION_TOKENS_SHA
from evaluate_resurface_more import pin_replay_backend, check_replay_backend, ARCHIVE_HASHES
from run_statequant import save_json

PROTOCOL = 'f18069d0340449316297077a2d912f7ef42fbf518436d9b311d8a731c96d722c'
KINDS = ('readout_tiers', 'dense3', 'dense3_equalized')
CACHE_BYTES = 28499968
ADAPTER = '7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0'
TABLE_KEYS = dict(readout_tiers='readout_permutations', dense3='original_permutations',
                  dense3_equalized='equalizer_exponents')

class CandidateInvalid(FloatingPointError):
    """Known candidate numerical/implementation/budget failure, not a GPU fault."""


def code_hashes():
    paths = list((ROOT/'mamba2_recall').glob('*.py'))
    paths += [Path(__file__), ROOT/'scripts/prepare_state_repair.py',
              ROOT/'scripts/evaluate_resurface_more.py', ROOT/'scripts/evaluate_quant_first.py',
              ROOT/'docs/STATE_REPAIR_PROTOCOL.md', ROOT/'docs/RESURFACE_MORE_BACKEND_REPLAY.md']
    return {str(p.relative_to(ROOT)):data.sha_file(p) for p in sorted(paths)}


def choose_candidate(results):
    baseline = results['old_sq_v2']
    p0, m0 = baseline['ppl']['ppl'], baseline['mk']['summary']['normal']['correct']
    valid = [kind for kind in KINDS if all(results[kind+suffix].get('complete') is True
             and math.isfinite(results[kind+suffix]['ppl']['ppl'])
             and results[kind+suffix]['cache']['total_bytes'] == CACHE_BYTES
             for suffix in ('_no_adapter','_v2'))]
    key = lambda kind:(results[kind+'_v2']['ppl']['ppl'],
                      -results[kind+'_v2']['mk']['summary']['normal']['correct'], KINDS.index(kind))
    eligible = [kind for kind in valid if results[kind+'_v2']['ppl']['ppl'] <= 1.01*p0
                and results[kind+'_v2']['mk']['summary']['normal']['correct'] >= m0-2]
    selected = min(eligible, key=key) if eligible else min(valid, key=key) if valid else None
    if selected and not eligible and results[selected+'_v2']['ppl']['ppl'] > 1.25*p0:
        selected = None
    return dict(selected=selected, eligible=eligible, valid=valid,
                screening_pass=bool(eligible), diagnostic_advancement=bool(selected and not eligible),
                stopped=selected is None, rule='Fixed protocol: eligible <=1.01x PPL and at most2 fewer TRAIN answers; otherwise finite best <=1.25x; PPL/MK/order tiebreak')


def make_execution(model, kind, old_permutations, repair):
    if kind == 's16':
        return StateQuant(model, 's16')
    if kind == 'old_sq':
        return StateQuant(model, 'sq3p25', old_permutations)
    if kind == 'readout_tiers':
        return StateQuant(model, 'sq3p25', repair['readout_permutations'])
    if kind == 'dense3':
        return DenseStateQuant(model, repair['original_permutations'], table_kind='permutation')
    if kind == 'dense3_equalized':
        return DenseStateQuant(model, repair['equalizer_exponents'], table_kind='equalizer')
    raise ValueError(kind)


def finite_cache(execution):
    tensors = [t for row in execution._cache for t in row.state.tensors.values()
               if t.is_floating_point()]
    # One synchronization checks actual stored scales/state, not only logits.
    conv=[row.conv for row in execution._cache]
    if not bool(torch.stack([torch.isfinite(t).all() for t in tensors+conv]).all()):
        raise FloatingPointError('Nonfinite persisted state or FP16 scale')
    return int(torch.stack([(t==0).sum() for t in tensors]).sum()) if execution.mode!='s16' else 0


def finite_exp(value):
    try: result=math.exp(value)
    except OverflowError as error: raise CandidateInvalid('PPL exponent overflow') from error
    if not math.isfinite(result):raise CandidateInvalid('Nonfinite PPL')
    return result


def validate_repair(payload,receipt,path,old):
    expected=dict(format='MAMBA2_STATE_REPAIR_CALIBRATION_V1',protocol_sha256=PROTOCOL,
        source_sha256=runtime.SOURCE_CHECKPOINT_SHA256,tokenizer_sha256=runtime.TOKENIZER_SHA256,
        train_file_sha256=TRAIN_SHA,prose_manifest_sha256=PROSE_MANIFEST_SHA,
        original_calibration_sha256='c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023',
        adapter=None,adapter_sha256=None,heldout_used=False,calibration_tokens=4096)
    if (any(payload.get(k)!=v or receipt.get(k)!=v for k,v in expected.items())
            or receipt.get('complete') is not True or receipt.get('fresh_source_no_adapter') is not True
            or receipt.get('sha256')!=data.sha_file(path) or receipt.get('bytes')!=path.stat().st_size
            or not torch.equal(payload['original_permutations'],old)):
        raise ValueError('Repair calibration source/provenance differs')
    proof=receipt['collection_forward_exactcheck'];checks=receipt['collector_checks']
    if (any(proof.get(k) is not True for k in ('complete','passed','hidden_bitwise_equal',
            'all56_ssm_carries_exact','all56_conv_caches_exact','mean_abs_statistics_exact'))
            or proof.get('probe_tokens')!=128 or checks.get('complete') is not True or checks.get('passed') is not True):
        raise ValueError('Collector original-forward equivalence proof missing')
    for hashes in (receipt['code_sha256'],checks['code_sha256']):
        for relative,digest in hashes.items():
            if data.sha_file(ROOT/relative)!=digest:raise ValueError('Calibration implementation changed: '+relative)
    import hashlib
    for key in TABLE_KEYS.values():
        table=payload[key]
        if (tuple(table.shape)!=(56,8,128) or table.numel()*table.element_size()!=57344
                or hashlib.sha256(table.contiguous().numpy().tobytes()).hexdigest()!=receipt['table_sha256'][key]):
            raise ValueError('Calibration table hash/geometry differs')
    stats=payload['statistics']
    if not bool((stats['sample_count_per_group']==4096*16*64).all()):raise ValueError('Calibration counts differ')
    for key in ('mean_abs','mean_readout_score'):
        if tuple(stats[key].shape)!=(56,8,128) or not bool(torch.isfinite(stats[key]).all()) or bool((stats[key]<0).any()):
            raise ValueError('Invalid calibration statistic')
    ranks=torch.argsort(stats['mean_readout_score'],dim=-1,descending=True,stable=True).to(torch.uint8)
    logs=stats['mean_abs'].double().clamp_min(1e-12).log2()
    exponents=(logs-logs.mean(-1,keepdim=True)).round().clamp(-8,8).to(torch.int8)
    if not torch.equal(ranks,payload['readout_permutations']) or not torch.equal(exponents,payload['equalizer_exponents']):
        raise ValueError('Calibration derivation differs')


@torch.inference_mode()
def evaluate(model, tokenizer, kind, old, repair, windows, cases, path, common):
    result = {**common, 'kind':kind, 'complete':False, 'ppl':{'windows':[]}, 'mk':{'rows':[]}}
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    expected = 122028032 if kind == 's16' else CACHE_BYTES
    with make_execution(model, kind, old, repair) as execution:
        # Every candidate must repeat its actual full-model128-token forward/cache.
        probe = windows[0][1][:128].cuda()[None]
        first = execution.backbone(probe, reset=True)
        if not bool(torch.isfinite(first).all()):raise CandidateInvalid('Nonfinite first probe')
        h0 = native.tensor_hash(first)
        zero_scales = finite_cache(execution)
        cache0 = {f'{i}.{k}':native.tensor_hash(t) for i,row in enumerate(execution._cache)
                  for k,t in {**row.state.tensors,'conv':row.conv}.items()}
        second = execution.backbone(probe, reset=True)
        if not bool(torch.isfinite(second).all()):raise CandidateInvalid('Nonfinite second probe')
        finite_cache(execution)
        if not torch.equal(first, second) or cache0 != {
                f'{i}.{k}':native.tensor_hash(t) for i,row in enumerate(execution._cache)
                for k,t in {**row.state.tensors,'conv':row.conv}.items()}:
            raise CandidateInvalid('Repeated-reset128-token hidden/cache differed')
        result['repeated_reset_probe'] = dict(tokens=128, hidden_sha256=h0,
            hidden_and_cache_exact=True, token_sha256=runtime.token_digest(probe.cpu().numpy()),
            cache_bytes=execution.cache_breakdown()['total_bytes'])
        if result['repeated_reset_probe']['cache_bytes'] != expected:
            raise CandidateInvalid('Candidate exceeds exact persistent cache budget')
        del first,second,probe,cache0
        nll=count=0
        for index,(start,window) in enumerate(windows):
            tokens=window.cuda()
            hidden=execution.backbone(tokens[:-1][None],reset=True)
            if not bool(torch.isfinite(hidden).all()):
                raise FloatingPointError('Nonfinite PPL hidden')
            zero_scales += finite_cache(execution)
            loss_sum=0.
            for pos in range(0,hidden.shape[1],64):
                end=min(pos+64,hidden.shape[1])
                logits=model.lm_head(hidden[:,pos:end]).float()
                loss=F.cross_entropy(logits.reshape(-1,256000),tokens[pos+1:end+1],reduction='sum')
                loss_sum+=float(loss)
                del logits,loss
            if not math.isfinite(loss_sum):
                raise FloatingPointError('Nonfinite PPL loss')
            targets=len(window)-1
            nll+=loss_sum;count+=targets
            result['ppl']['windows'].append(dict(start=start,target_tokens=targets,
                token_sha256_int64le=runtime.token_digest(window.numpy()),nll=loss_sum,ppl=finite_exp(loss_sum/targets)))
            result['ppl'].update(nll=nll,target_tokens=count,ppl=finite_exp(nll/count))
            result['cache']=execution.cache_breakdown()
            save_json(path,result)
            if index==0 or (index+1)%8==0 or index+1==len(windows):
                print(f'[{common["arm"]} PPL] {index+1}/{len(windows)} ppl={result["ppl"]["ppl"]:.6f}',flush=True)
            del hidden,tokens
        result['ppl']['elapsed_seconds']=time.time()-started
        mk_start=time.time()
        for index,case in enumerate(cases):
            encoded=tokenizer.encode(case['prompt'])
            hidden=execution.backbone(torch.tensor(encoded,device='cuda',dtype=torch.long)[None],reset=True)[:,-1:]
            generated=[]
            for step in range(12):
                logits=model.lm_head(hidden)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('Nonfinite MK logits')
                token=int(logits.argmax(-1).item());generated.append(token)
                if token==tokenizer.eos_token_id or step==11:break
                hidden=execution.backbone(torch.tensor([[token]],device='cuda'))
            zero_scales+=finite_cache(execution)
            output=tokenizer.decode(generated)
            match=re.search(r'(?<!\d)\d{6}(?!\d)',output)
            prediction=match.group() if match else None
            result['mk']['rows'].append({**case,'prompt_tokens':len(encoded),
                'prompt_token_sha256_int64le':runtime.token_digest(encoded),'generated_ids':generated,
                'output':output,'prediction':prediction,'correct':prediction==case['answer']})
            if index==0 or (index+1)%8==0:
                save_json(path,result)
                print(f'[{common["arm"]} MK] {index+1}/{len(cases)}',flush=True)
        result['mk']['summary']={}
        for condition in ('normal','target_removed'):
            rows=[r for r in result['mk']['rows'] if r['condition']==condition]
            correct=sum(r['correct'] for r in rows)
            result['mk']['summary'][condition]=dict(correct=correct,count=len(rows),accuracy=correct/len(rows) if rows else None)
        result['mk']['elapsed_seconds']=time.time()-mk_start
        result['cache']=execution.cache_breakdown()
        if result['cache']['total_bytes']!=expected:raise CandidateInvalid('Final cache budget changed')
        result['persistent_float_finite_checks_passed']=True
        result['zero_scale_observations']=zero_scales
        result['zero_scale_scope']='Repeated final-cache observations; includes true zero states and FP16 scale underflow; not unique underflow events'
        result['gpu_memory']=runtime.gpu_memory_receipt()
    result.update(complete=True,elapsed_seconds=time.time()-started)
    save_json(path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('screen','full'),required=True)
    for name in ('source-dir','calibration','repair-calibration','parent-training-report','prose-tokens','codec-checks','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--screening-report',type=Path)
    parser.add_argument('--parent-eval-dir',type=Path)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve evidence: require a fresh output directory')
    if data.sha_file(ROOT/'docs/STATE_REPAIR_PROTOCOL.md')!=PROTOCOL:raise ValueError('Frozen protocol changed')
    torch.set_num_threads(8);torch.manual_seed(20260928);torch.cuda.manual_seed_all(20260928)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    old,old_receipt,parent,adapter_path=load_candidate(SimpleNamespace(calibration=args.calibration,
        training_report=args.parent_training_report),tokenizer)
    if parent['adapter']['sha256']!=ADAPTER:raise ValueError('Wrong frozen parent adapter')
    repair=torch.load(args.repair_calibration,map_location='cpu',weights_only=True)
    receipt=json.loads(args.repair_calibration.with_suffix('.json').read_text())
    validate_repair(repair,receipt,args.repair_calibration,old)
    codec_checks=json.loads(args.codec_checks.read_text())
    if (codec_checks.get('format')!='MAMBA2_DENSE3_CODEC_CHECK_V1' or codec_checks.get('complete') is not True
            or 'error' in codec_checks or not codec_checks.get('checks')
            or any(row.get('pass') is not True for row in codec_checks['checks'])):
        raise ValueError('Completed dense codec checks required')
    for relative,digest in codec_checks['code_sha256'].items():
        if data.sha_file(ROOT/relative)!=digest:raise ValueError('Dense codec changed after checks')
    table_hashes={k:native.tensor_hash(repair[v]) for k,v in TABLE_KEYS.items()}
    for kind in KINDS:
        table=repair[TABLE_KEYS[kind]]
        if tuple(table.shape)!=(56,8,128) or table.numel()*table.element_size()!=57344:
            raise ValueError('Invalid metadata budget')
    hashes=code_hashes();policy=pin_replay_backend()
    if args.stage=='screen':
        train=load_train_tokens(args.prose_tokens)
        windows=[(i*2048,train[i,:512]) for i in range(8,16)]
        cases=[c for c in data.generate_cases('train') if c['sample'] in range(0,256,32)]
        if len(cases)!=48:raise ValueError('TRAIN screen count changed')
        dataset=dict(split='train',file_sha256=data.sha_file(args.prose_tokens),rows=list(range(8,16)),tokens_per_row=512)
        arms=[('source_s16','s16',False),('old_sq_no_adapter','old_sq',False),('old_sq_v2','old_sq',True)]
        arms += [(kind+suffix,kind,adapted) for kind in KINDS for suffix,adapted in (('_no_adapter',False),('_v2',True))]
        arms.append(('restored_old_sq_v2','old_sq',True))
        screen=None
    else:
        if args.screening_report is None or args.parent_eval_dir is None:raise ValueError('Full requires frozen screening and archived parent')
        screen=json.loads(args.screening_report.read_text())
        if (screen.get('complete') is not True or screen.get('protocol_sha256')!=PROTOCOL
                or screen.get('repair_calibration_sha256')!=data.sha_file(args.repair_calibration)
                or screen.get('codec_checks_sha256')!=data.sha_file(args.codec_checks)
                or screen.get('code_hashes')!=hashes or not screen.get('restoration',{}).get('complete')):
            raise ValueError('Screening provenance/code/replay differs')
        screened={}
        for arm,digest in screen['report_sha256'].items():
            path=args.screening_report.parent/f'screen_{arm}.json'
            if data.sha_file(path)!=digest:raise ValueError('Screening arm changed')
            screened[arm]=json.loads(path.read_text())
        selection=choose_candidate(screened)
        if selection!=screen['selection'] or selection['selected'] is None:raise ValueError('No fixed selected candidate')
        chosen=selection['selected']
        ids,dataset=load_wikitext_tokens(tokenizer,'validation');windows=ppl_windows(ids,2048)
        cases=data.generate_cases('confirm')
        if (len(windows)!=130 or sum(len(w)-1 for _,w in windows)!=264764
                or dataset['token_stream_sha256_int64le']!=VALIDATION_TOKENS_SHA or len(cases)!=768):
            raise ValueError('Full population changed')
        arms=[('old_sq_v2','old_sq',True),(chosen+'_v2',chosen,True),('restored_old_sq_v2','old_sq',True)]
    args.out.mkdir(parents=True)
    model=runtime.load_source_model(args.source_dir);frozen=FrozenBase(model)
    common=dict(format='MAMBA2_STATE_REPAIR_EVAL_V1',stage=args.stage,protocol_sha256=PROTOCOL,
        source_checkpoint_sha256=runtime.SOURCE_CHECKPOINT_SHA256,tokenizer_sha256=tokenizer.sha256,
        original_calibration_sha256=data.sha_file(args.calibration),repair_calibration_sha256=data.sha_file(args.repair_calibration),
        repair_calibration_receipt_sha256=data.sha_file(args.repair_calibration.with_suffix('.json')),
        parent_training_report_sha256=data.sha_file(args.parent_training_report),parent_adapter_sha256=ADAPTER,
        table_sha256=table_hashes,dataset=dataset,environment=runtime.environment_receipt(),backend_policy=policy,
        code_hashes=hashes,adapter_binding_scope='Original frozen v2 export binding retained; candidate codec/table override is explicit and untrained',
        codec_checks_sha256=data.sha_file(args.codec_checks),
        selection_scope='TRAIN-only fixed screening; validation and CONFIRM only after selection')
    results={};started=time.time()
    for arm,kind,adapted in arms:
        print('[state-repair arm] '+arm,flush=True)
        path=args.out/f'{args.stage}_{arm}.json'
        with native.install_fp16(model,adapter_path,expected_binding=parent['binding']) if adapted else contextlib.nullcontext() as bank:
            frozen.check()
            if bank and {k:native.tensor_hash(v) for k,v in bank.masters.items()}!=parent['adapter']['tensor_sha256']:
                raise ValueError('Installed adapter differs')
            try:
                result=evaluate(model,tokenizer,kind,old,repair,windows,cases,path,
                    {**common,'arm':arm,'adapter_sha256':ADAPTER if adapted else None})
            except FloatingPointError as error:
                if args.stage!='screen' or kind not in KINDS:raise
                result=json.loads(path.read_text()) if path.exists() else {**common,'arm':arm,'kind':kind,'adapter_sha256':ADAPTER if adapted else None}
                result.update(complete=False,error=str(error),excluded_from_selection=True)
            result['frozen_source']=frozen.check()
            result['backend_policy_check']=check_replay_backend(policy)
            result['adapter_content_unchanged']=bank is None or {k:native.tensor_hash(v) for k,v in bank.masters.items()}==parent['adapter']['tensor_sha256']
            if not result['adapter_content_unchanged']:raise RuntimeError('Adapter changed during inference')
            save_json(path,result);results[arm]=result
        frozen.check()
        if args.stage=='full' and arm=='old_sq_v2':
            archive_path=args.parent_eval_dir/'full_resurface_sq3p25.json'
            if data.sha_file(archive_path)!=ARCHIVE_HASHES['resurface_sq3p25']:raise ValueError('Archived parent differs')
            parent_replay=check_restoration(json.loads(archive_path.read_text()),result)
            save_json(args.out/'full_parent_replay.json',parent_replay)
    restoration=check_restoration(results['old_sq_v2'],results['restored_old_sq_v2'])
    restoration['scope']='Restore original codec/table with the same frozen v2 adapter; all windows and generated IDs exact'
    save_json(args.out/f'{args.stage}_restoration.json',restoration)
    outcome=dict(format='MAMBA2_STATE_REPAIR_COMPARISON_V1',complete=True,stage=args.stage,protocol_sha256=PROTOCOL,
        repair_calibration_sha256=data.sha_file(args.repair_calibration),parent_adapter_sha256=ADAPTER,
        code_hashes=hashes,backend_policy=policy,restoration=restoration,
        codec_checks_sha256=data.sha_file(args.codec_checks),
        report_sha256={arm:data.sha_file(args.out/f'{args.stage}_{arm}.json') for arm in results},
        elapsed_seconds=time.time()-started)
    if args.stage=='screen':
        outcome['selection']=choose_candidate(results)
    else:
        paired=compare_pair(results['old_sq_v2'],results[chosen+'_v2'])
        archive_path=args.parent_eval_dir/'full_source_s16.json'
        if data.sha_file(archive_path)!=ARCHIVE_HASHES['source_s16']:raise ValueError('Archived S16 differs')
        gap=compare_pair(json.loads(archive_path.read_text()),results[chosen+'_v2'])
        checks=dict(ppl_at_least_1pct_better=paired['ppl_relative_change']<=-.01,
            mk_no_observed_decrease=paired['normal_mk_correct_delta']>=0,
            mk_95ci_lower_at_least_minus2pp=paired['normal_mk_paired_bootstrap_95ci'][0]>=-.02,
            cache_same_budget=results[chosen+'_v2']['cache']['total_bytes']==CACHE_BYTES)
        outcome.update(selected=chosen,selection=screen['selection'],screening_report_sha256=data.sha_file(args.screening_report),
            comparison=paired,remaining_gap_vs_archived_s16=gap,gate_checks=checks,repair_gate_pass=all(checks.values()),
            original_ppl_restored_within_1pct=gap['ppl_relative_change']<=.01,parent_replay=parent_replay)
    save_json(args.out/f'{args.stage}_comparison.json',outcome)
    print(json.dumps(outcome,indent=2),flush=True)


if __name__=='__main__':main()
