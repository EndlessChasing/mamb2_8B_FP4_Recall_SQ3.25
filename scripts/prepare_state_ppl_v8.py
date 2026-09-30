#!/usr/bin/env python3
"""Propose static layer mixtures using unadapted TRAIN PPL, never heldout/MK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mamba2_recall import runtime, resurface_data as data
from prepare_state_first_v5 import KINDS, check_table, need, read_json, tensor_sha, write_payload
from prepare_quant_first import write_new_json
import run_state_ppl_v6 as v6

PROTOCOL_SHA = '880839c0b0919d0a9a119647d2d9c1657e5b8eb5389c8c2ee22ee1785917aa01'
V6_SELECTED_SHA = '098930d1af5e5821b277640d236f7607c6428b48d36f7117e2ea4aa87e656303'
BASELINE_KIND = 'preserve_int8'
BASELINE_POLICY = 'preserve_int8__stored_scale'
ALTERNATIVES = tuple(kind for kind in KINDS if kind != BASELINE_KIND)
CALIBRATION_ROWS = tuple(range(72,80))
SCREEN_ROWS = tuple(range(112,144))
TOP_K = (1,2,4,8,16)
ARM_SPECS = [('baseline', None, None)] + [
    (f'layer{layer:02d}_{kind}', layer, kind) for layer in range(56) for kind in ALTERNATIVES
] + [('restored_baseline', None, None)]
ARMS = tuple(arm for arm,_,_ in ARM_SPECS)
EVAL_FORMAT = 'MAMBA2_STATE_PPL_LAYER_EVAL_V1'
COMPARISON_FORMAT = 'MAMBA2_STATE_PPL_LAYER_CALIBRATION_V1'
CANDIDATES_FORMAT = 'MAMBA2_STATE_PPL_LAYER_CANDIDATES_V1'


def code_hashes():
    return dict(v6.code_hashes(), **{
        'scripts/prepare_state_ppl_v8.py': data.sha_file(__file__),
        'docs/STATE_PPL_V8_PROTOCOL.md': data.sha_file(ROOT/'docs/STATE_PPL_V8_PROTOCOL.md')})


def check_protocol():
    need(data.sha_file(ROOT/'docs/STATE_PPL_V8_PROTOCOL.md') == PROTOCOL_SHA,
         'Frozen v8 protocol changed or is not yet bound')


def load_input_artifacts(candidates_path, v5_selected_path, v6_selected_path, codec_checks_path, prose_tokens_path):
    check_protocol()
    candidates, old_binding, old_windows, _ = v6.load_inputs(candidates_path, v5_selected_path,
        codec_checks_path, prose_tokens_path)
    need(data.sha_file(v6_selected_path) == V6_SELECTED_SHA, 'Pinned v6 selected artifact differs')
    selected, _, comparison = v6.load_selected_calibration(v6_selected_path, candidates, old_binding, old_windows)
    need(selected['selected_id'] == BASELINE_POLICY and comparison['selection']['stopped'] is False,
         'Layer-calibration baseline must be the frozen v6 stored-scale selection')
    train = v6.load_train_tokens(prose_tokens_path)
    windows = [(row*2048,train[row,:2048]) for row in CALIBRATION_ROWS]
    need(len(windows) == 8 and sum(len(window)-1 for _,window in windows) == 16376,
         'Layer calibration TRAIN population differs')
    binding = dict(protocol_sha256=PROTOCOL_SHA, v6_input_binding=old_binding,
        v6_selected_calibration_sha256=V6_SELECTED_SHA,
        v6_selected_receipt_sha256=data.sha_file(v6_selected_path.with_suffix('.json')),
        v6_selection_report_sha256=selected['selection_report_sha256'],
        baseline_policy=BASELINE_POLICY, baseline_table_sha256=selected['table_sha256'],
        scale_mode='stored_scale', int4_clip=1., adapter_loaded=False, adapter_sha256=None,
        heldout_used=False, mk_used=False)
    return candidates, binding, windows


def table_for_arm(tables, layer, kind):
    table = tables[BASELINE_KIND].clone().contiguous()
    if layer is not None:
        need(type(layer) is int and 0 <= layer < 56 and kind in ALTERNATIVES, 'Invalid fixed layer swap')
        table[layer].copy_(tables[kind][layer])
    else:
        need(kind is None, 'Baseline cannot name a replacement table')
    check_table(table)
    return table


def execution_spec(arm):
    spec = v6.candidate_spec(BASELINE_POLICY)
    spec.update(candidate_id=arm, candidate_name='static_layer_mix')
    return spec


def rank_single_layer(results):
    """Select a strictly improving alternative per layer; effects are not additive."""
    baseline = results['baseline']
    need(v6.valid_candidate(baseline), 'Valid fixed baseline required')
    nll0 = baseline['ppl']['nll']
    layers = []
    for layer in range(56):
        options = []
        for kind in ALTERNATIVES:
            arm = f'layer{layer:02d}_{kind}'
            row = results[arm]
            valid = v6.valid_candidate(row)
            options.append(dict(arm=arm, kind=kind, valid=valid,
                nll=row['ppl']['nll'] if valid else None,
                delta_nll=row['ppl']['nll']-nll0 if valid else None,
                ppl=row['ppl']['ppl'] if valid else None))
        valid_options = [row for row in options if row['valid']]
        best = min(valid_options, key=lambda row:(row['nll'],ALTERNATIVES.index(row['kind']))) if valid_options else None
        layers.append(dict(layer=layer, alternatives=options, best_alternative=best,
            improving=best is not None and best['nll'] < nll0))
    ranked = [dict(layer=row['layer'], **row['best_alternative']) for row in layers if row['improving']]
    ranked.sort(key=lambda row:(row['delta_nll'],row['layer'],ALTERNATIVES.index(row['kind'])))
    return dict(baseline_nll=nll0, baseline_ppl=baseline['ppl']['ppl'], layers=layers,
        ranked_improving_swaps=ranked, improving_layers=len(ranked), stopped=not ranked,
        calibration_rows=list(CALIBRATION_ROWS), future_screen_rows=list(SCREEN_ROWS),
        rule='Per layer minimum finite complete alternative NLL; fixed alternative order ties; strictly negative delta only; rank(deltaNLL,layer,alternative order)',
        interpretation='Single-layer deltas propose combinations; they do not predict additive combined gains',
        adapter_used=False, heldout_used=False, mk_used=False)


def derive_candidates(tables, ranking):
    """Baseline plus fixed top-k proposals, deduplicated by actual table content."""
    ranked = ranking['ranked_improving_swaps']
    requests = [('baseline',0)] + [(f'top{k}',min(k,len(ranked))) for k in TOP_K] + [('allnegative',len(ranked))]
    unique, inventory, proposals, seen = {}, {}, [], {}
    for name, count in requests:
        table = tables[BASELINE_KIND].clone().contiguous()
        swaps = [dict(layer=row['layer'], kind=row['kind'], calibration_delta_nll=row['delta_nll'],
                      calibration_arm=row['arm']) for row in ranked[:count]]
        for swap in swaps:
            table[swap['layer']].copy_(tables[swap['kind']][swap['layer']])
        check_table(table)
        digest = tensor_sha(table)
        kept = seen.get(digest)
        if kept is None:
            kept = name
            seen[digest] = name
            unique[name] = table
            inventory[name] = dict(table_sha256=digest, swap_count=count, swaps=swaps,
                scale_mode='stored_scale',int4_clip=1.,cache_bytes=v6.CACHE_BYTES)
        else:
            need(torch.equal(unique[kept],table), 'Table hash collision')
        proposals.append(dict(requested_name=name, requested_top_k='all' if name == 'allnegative'
            else 0 if name == 'baseline' else int(name[3:]), effective_swap_count=count,
            retained_name=kept, duplicate_of=kept if kept != name else None,table_sha256=digest))
    return unique, inventory, proposals


def export_candidates(out, tables, binding, ranking, comparison):
    derived, inventory, proposals = derive_candidates(tables,ranking)
    common = dict(protocol_sha256=PROTOCOL_SHA,input_binding=binding,
        calibration_report_sha256=data.sha_file(out/'calibration_comparison.json'),
        candidate_order=list(derived),candidate_specs=inventory,proposals=proposals,
        calibration_stopped=ranking['stopped'],baseline_name='baseline',scale_mode='stored_scale',int4_clip=1.,
        runtime_table_bytes=57344,cache_bytes=v6.CACHE_BYTES,
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,mk_used=False,
        candidate_scope='Unmeasured combinations proposed by TRAIN single-layer deltas; separate TRAIN screen required')
    payload = dict(format=CANDIDATES_FORMAT,**common,tables=derived)
    path = out/'candidates.pt'
    write_payload(path,payload)
    receipt = dict(format=CANDIDATES_FORMAT,**common,complete=True,file=path.name,
        sha256=data.sha_file(path),bytes=path.stat().st_size,code_hashes=code_hashes())
    write_new_json(path.with_suffix('.json'),receipt)
    return receipt


def validate_calibration(directory, candidates, binding, windows):
    """Hash-check every raw arm and reconstruct the fixed proposal arithmetic."""
    directory = Path(directory)
    comparison = read_json(directory/'calibration_comparison.json')
    need(comparison.get('format') == COMPARISON_FORMAT and comparison.get('complete') is True
         and comparison.get('protocol_sha256') == PROTOCOL_SHA and comparison.get('input_binding') == binding
         and comparison.get('code_hashes') == code_hashes() and comparison.get('adapter_loaded') is False
         and comparison.get('heldout_used') is False and comparison.get('mk_used') is False
         and tuple(comparison['report_sha256']) == ARMS, 'Layer calibration comparison binding differs')
    identities = v6.window_identity(windows)
    results = {}
    for arm,layer,kind in ARM_SPECS:
        path = directory/('calibration_'+arm+'.json')
        need(data.sha_file(path) == comparison['report_sha256'][arm], 'Raw calibration arm changed: '+arm)
        row = read_json(path)
        table = table_for_arm(candidates['tables'],layer,kind)
        need(row.get('format') == EVAL_FORMAT and row.get('stage') == 'calibration'
             and row.get('arm') == arm and row.get('input_binding') == binding
             and row.get('code_hashes') == code_hashes() and row.get('dataset') == calibration_dataset()
             and row.get('layer_replacement') == dict(layer=layer,kind=kind)
             and row.get('candidate_table_sha256') == tensor_sha(table)
             and all(row.get(key) == value for key,value in execution_spec(arm).items())
             and row.get('adapter_loaded') is False and row.get('adapter_sha256') is None
             and row.get('heldout_used') is False and row.get('heldout_used_for_selection') is False
             and row.get('mk_used') is False, 'Calibration arm identity differs: '+arm)
        if row.get('complete'):
            v6.check_ppl(row,identities)
            need(v6.valid_candidate(row), 'Complete calibration arm integrity differs')
        else:
            need(layer is not None and row.get('excluded_from_selection') is True
                 and row.get('error_type') == 'CandidateInvalid' and not row.get('fatal_failure',False),
                 'Unrecognized calibration failure')
        results[arm] = row
    ranking = rank_single_layer(results)
    need(comparison['ranking'] == ranking, 'Single-layer ranking differs')
    restoration = v6.check_restoration(results['baseline'],results['restored_baseline'])
    need(comparison['restoration'] == restoration and read_json(directory/'calibration_restoration.json') == restoration,
         'Layer calibration baseline restoration differs')
    return comparison,ranking


def load_layer_candidates(path, candidates_path, v5_selected_path, v6_selected_path, codec_checks_path, prose_tokens_path):
    path = Path(path)
    original,binding,windows = load_input_artifacts(candidates_path,v5_selected_path,v6_selected_path,
        codec_checks_path,prose_tokens_path)
    comparison,ranking = validate_calibration(path.parent,original,binding,windows)
    payload = torch.load(path,map_location='cpu',weights_only=True)
    receipt = read_json(path.with_suffix('.json'))
    derived,inventory,proposals = derive_candidates(original['tables'],ranking)
    need(payload.get('format') == receipt.get('format') == CANDIDATES_FORMAT
         and receipt.get('complete') is True and receipt.get('file') == path.name
         and receipt.get('sha256') == data.sha_file(path) and receipt.get('bytes') == path.stat().st_size,
         'Layer candidate payload/receipt identity differs')
    expected = dict(protocol_sha256=PROTOCOL_SHA,input_binding=binding,
        calibration_report_sha256=data.sha_file(path.parent/'calibration_comparison.json'),
        candidate_order=list(derived),candidate_specs=inventory,proposals=proposals,
        calibration_stopped=ranking['stopped'],baseline_name='baseline',scale_mode='stored_scale',int4_clip=1.,
        runtime_table_bytes=57344,cache_bytes=v6.CACHE_BYTES,adapter_loaded=False,adapter_sha256=None,
        heldout_used=False,mk_used=False,
        candidate_scope='Unmeasured combinations proposed by TRAIN single-layer deltas; separate TRAIN screen required')
    need(all(payload.get(key) == receipt.get(key) == value for key,value in expected.items())
         and receipt.get('code_hashes') == code_hashes() and tuple(payload['tables']) == tuple(derived),
         'Layer candidate derivation/provenance differs')
    for name,table in derived.items():
        check_table(payload['tables'][name])
        need(torch.equal(payload['tables'][name],table), 'Exported combined table differs: '+name)
    return payload,receipt,comparison


def calibration_dataset():
    return dict(split='train',file_sha256=v6.TRAIN_SHA,rows=list(CALIBRATION_ROWS),
        tokens_per_row=2048,windows=8,target_tokens=16376)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','candidates','v5-selected-calibration','v6-selected-calibration',
                 'prose-tokens','codec-checks','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    need(not args.out.exists(), 'Fresh output directory required; preserve prior evidence')
    candidates,binding,windows = load_input_artifacts(args.candidates,args.v5_selected_calibration,
        args.v6_selected_calibration,args.codec_checks,args.prose_tokens)
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256 == runtime.TOKENIZER_SHA256, 'Tokenizer differs')
    torch.set_num_threads(8)
    torch.manual_seed(20260928);torch.cuda.manual_seed_all(20260928)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    policy = v6.pin_replay_backend()
    args.out.mkdir(parents=True)
    model = runtime.load_source_model(args.source_dir)
    frozen = v6.FrozenBase(model)
    need(v6.no_adapter_hooks(model), 'Unexpected adapter hooks on source')
    common = dict(format=EVAL_FORMAT,stage='calibration',protocol_sha256=PROTOCOL_SHA,input_binding=binding,
        dataset=calibration_dataset(),environment=runtime.environment_receipt(),code_hashes=code_hashes(),
        backend_policy=policy,adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=False,
        heldout_used_for_selection=False,
        selection_scope='Single-layer full-model TRAIN PPL proposes static combinations; later disjoint TRAIN screen chooses one')
    results = {}
    started = time.time()
    identities = v6.window_identity(windows)
    for arm,layer,kind in ARM_SPECS:
        print('[layer-calibration-v8 arm] '+arm,flush=True)
        need(v6.no_adapter_hooks(model), 'Unexpected adapter hooks during calibration')
        frozen.check()
        table = table_for_arm(candidates['tables'],layer,kind)
        digest = tensor_sha(table)
        spec = execution_spec(arm)
        path = args.out/('calibration_'+arm+'.json')
        arm_common = dict(common,arm=arm,candidate_table_sha256=digest,
            layer_replacement=dict(layer=layer,kind=kind))
        try:
            result = v6.evaluate(model,table,spec,windows,path,arm_common)
        except v6.CandidateInvalid as error:
            result = read_json(path)
            result.update(complete=False,error=repr(error),error_type='CandidateInvalid')
            if layer is None:
                result['fatal_failure'] = True;v6.save_json(path,result)
                raise
            result['excluded_from_selection'] = True
        except Exception as error:
            result = read_json(path) if path.exists() else dict(arm_common,**spec)
            result.update(complete=False,error=repr(error),error_type=type(error).__name__,fatal_failure=True)
            v6.save_json(path,result)
            raise
        result['frozen_source'] = frozen.check()
        result['backend_policy_check'] = v6.check_replay_backend(policy)
        need(tensor_sha(table) == digest, 'CPU candidate table mutated')
        result['candidate_table_unchanged'] = True
        need(v6.no_adapter_hooks(model), 'Adapter hooks changed during calibration')
        if result['complete']:
            v6.check_ppl(result,identities)
        v6.save_json(path,result)
        results[arm] = result
    restoration = v6.check_restoration(results['baseline'],results['restored_baseline'])
    v6.save_json(args.out/'calibration_restoration.json',restoration)
    ranking = rank_single_layer(results)
    comparison = dict(format=COMPARISON_FORMAT,complete=True,protocol_sha256=PROTOCOL_SHA,
        stage='calibration',input_binding=binding,code_hashes=code_hashes(),backend_policy=policy,
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,mk_used=False,ranking=ranking,
        restoration=restoration,report_sha256={arm:data.sha_file(args.out/('calibration_'+arm+'.json')) for arm in ARMS},
        elapsed_seconds=time.time()-started)
    v6.save_json(args.out/'calibration_comparison.json',comparison)
    receipt = export_candidates(args.out,candidates['tables'],binding,ranking,comparison)
    load_layer_candidates(args.out/'candidates.pt',args.candidates,args.v5_selected_calibration,
        args.v6_selected_calibration,args.codec_checks,args.prose_tokens)
    print(json.dumps(dict(complete=True,calibration_stopped=ranking['stopped'],improving_layers=ranking['improving_layers'],
        candidates=receipt['candidate_order'],candidates_sha256=receipt['sha256'],candidate_specs=receipt['candidate_specs']),indent=2),flush=True)


if __name__ == '__main__':
    main()
