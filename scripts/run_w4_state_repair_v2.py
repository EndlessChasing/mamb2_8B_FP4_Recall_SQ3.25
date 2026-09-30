#!/usr/bin/env python3
"""TRAIN-select one same-byte unadapted W4 state policy, then confirm PPL."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime, resurface_native as native
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from prepare_quant_first import load_train_tokens
from prepare_state_first_v5 import write_payload
from evaluate_quant_first import FrozenBase
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from run_statequant import save_json
from run_state_ppl_v6 import CandidateInvalid, finite_cache, cache_identity, finite_exp, no_adapter_hooks
import w4_state_binding_v1 as base
import w4_state_repair_binding_v2 as binding
import w4_state_repair_codec_v2 as codec

need, sha, read_json = base.need, base.sha, base.read_json
FORMAT = 'W4_STATE_REPAIR_V2_EVAL_V1'
SCREEN_FORMAT = 'W4_STATE_REPAIR_V2_SCREEN_V1'
COMPARE_FORMAT = 'W4_STATE_REPAIR_V2_COMPARISON_V1'
SELECTION_FORMAT = 'W4_STATE_REPAIR_V2_SELECTION_V1'
SCREEN_ROWS = tuple(range(256, 288))
RESTORED = 'restored_baseline'
FULL_ARMS = ('baseline', 'selected', RESTORED)
EXPECTED_BASELINE_PPL = 9.103235516433815
CACHE_BYTES = 28499968


def code_hashes():
    return binding.code_hashes(extra=['scripts/run_w4_state_repair_v2.py',
        'scripts/w4_state_repair_codec_v2.py'])


def check_source_inventory(inventory):
    need(isinstance(inventory,dict) and inventory, 'Executed source inventory missing')
    for name,digest in inventory.items():
        path = (ROOT/name).resolve()
        need(not Path(name).is_absolute() and path.is_relative_to(ROOT.resolve())
             and path.is_file() and sha(path) == digest, 'Executed source changed: '+name)


def native_snapshot(model):
    return [(module,module.forward,dict(module._forward_pre_hooks),dict(module._forward_hooks))
            for layer in model.backbone.layers for module in (layer.mixer,layer.mixer.norm)]


def check_native_snapshot(snapshot):
    need(all(module.forward == forward and dict(module._forward_pre_hooks) == pre
             and dict(module._forward_hooks) == post for module,forward,pre,post in snapshot),
         'State controller failed to restore native forwards/hooks')
    return dict(complete=True,native_mixer_forwards_restored=True,native_hooks_restored=True)


def require_cache(execution,tokens=None):
    actual = execution.cache_breakdown()
    expected = dict(mode='sq3p25',scale_mode='stored_scale',int4_clip=1.,diagnostic=None,
        is_3p25_candidate=True,batch_size=1,allocated_layers=56,conv_fp16_bytes=4587520,
        ssm_payload_bytes=22020096,ssm_scale_bytes=1835008,ssm_total_bytes=23855104,
        permutation_bytes=57344,total_bytes=CACHE_BYTES,calibration_workspace_bytes=0,
        diagnostic_dense_fp32_bytes=0,row_bytes=52)
    if tokens is not None: expected['tokens_per_layer'] = [tokens]*56
    need(all(actual.get(k) == v for k,v in expected.items()), 'Physical recurrent cache budget/geometry differs')
    need(len({entry.tokens for entry in execution._cache}) == 1, 'Per-layer cache positions differ')
    return actual


def checked_descriptor(execution,layouts):
    need(len(layouts) == 56 and len(execution._cache) == 56, 'Exactly 56 layer layouts required')
    for entry,layout in zip(execution._cache,layouts):
        codec.validate_state(entry.state,execution.device,layout)
        need(list(entry.conv.shape) == [1,10240,4] and entry.conv.dtype == torch.float16
             and entry.conv.untyped_storage().nbytes() == 81920, 'Convolution physical layout differs')
    result = execution.storage_descriptor()
    need(result.get('layouts') == list(layouts) and result.get('row_bytes') == 52
         and result.get('payload_bytes') == 48 and result.get('scale_bytes') == 4
         and result.get('resident_layout_metadata_bytes') == 0 and len(result.get('layers',[])) == 56,
         'State policy storage descriptor differs')
    for row,entry,layout in zip(result['layers'],execution._cache,layouts):
        n8,n4,nzero = (int(value) for value in layout.split('_'))
        need(n8+n4+nzero == 128 and n8+n4//2+4 == 52
             and row.get('layout') == layout and row.get('state_bytes') == 128*64*52,
             'Per-layer state allocation differs')
        expected_widths = dict(lo=n8//2,hi=n8//2,q4=n4//2)
        for key,width in expected_widths.items():
            need(row['tensors'][key] == dict(shape=[1,128,64,width],dtype='torch.uint8',storage_bytes=128*64*width),
                 'Packed tensor width/storage differs: '+key)
        for key in ('s8','s4'):
            need(row['tensors'][key] == dict(shape=[1,128,64],dtype='torch.float16',storage_bytes=16384),
                 'FP16 scale tensor storage differs')
    return result


def candidate_metadata(candidate):
    layouts = tuple(candidate['layouts'])
    need(len(layouts) == 56 and all(layout in binding.LAYOUT_ORDER for layout in layouts),
         'Candidate must explicitly specify 56 declared layouts')
    digest = base.check_table(candidate['table'])
    return dict(candidate_id=candidate['id'],candidate_table_name=candidate['table_name'],
        candidate_table_sha256=digest,layer_layouts=list(layouts),scale_mode='stored_scale',int4_clip=1.,
        deployable=True,diagnostic=None,adapter_loaded=False,adapter_sha256=None,mk_used=False)


@contextlib.contextmanager
def execution_for(model,candidate):
    snapshot = native_snapshot(model)
    digest = base.check_table(candidate['table'])
    try:
        with codec.StateRepairV2(model,candidate['table'],tuple(candidate['layouts'])) as execution:
            try:
                yield execution
            finally:
                need(tuple(execution.layouts) == tuple(candidate['layouts']), 'Actual per-layer layout policy mutated')
                need(native.tensor_hash(execution.permutations) == digest, 'Actual runtime coordinate table mutated')
    finally:
        check_native_snapshot(snapshot)


def window_identity(windows):
    return [dict(start=start,target_tokens=len(window)-1,
        token_sha256_int64le=runtime.token_digest(window.numpy())) for start,window in windows]


def check_ppl(result,windows):
    need(result.get('complete') is True and 'error' not in result and 'mk' not in result,
         'Complete finite PPL-only evaluation required')
    expected = window_identity(windows); actual = result['ppl']['windows']
    need(len(actual) == len(expected) and all(all(row.get(k) == v for k,v in identity.items())
         for row,identity in zip(actual,expected)), 'PPL window/token population differs')
    total = 0.; count = 0
    for row in actual:
        need(math.isfinite(row['nll']) and row['nll'] >= 0 and row['target_tokens'] > 0
             and row['ppl'] == finite_exp(row['nll']/row['target_tokens']), 'Window NLL arithmetic differs')
        total += row['nll']; count += row['target_tokens']
    need(result['ppl']['nll'] == total and result['ppl']['target_tokens'] == count
         and result['ppl']['ppl'] == finite_exp(total/count), 'Aggregate PPL arithmetic differs')


@torch.inference_mode()
def evaluate(model,candidate,windows,path,common):
    """Generic 56-layout evaluator reusable by a separately frozen future family."""
    result = dict(common,**candidate_metadata(candidate),complete=False,ppl=dict(windows=[]))
    save_json(path,result); started = time.time()
    torch.cuda.reset_peak_memory_stats()
    with execution_for(model,candidate) as execution:
        # Record the actual 56-layer allocation even if the first candidate forward is nonfinite.
        execution.reset(1)
        result['allocated_cache'] = require_cache(execution,0)
        result['storage_descriptor_initial'] = checked_descriptor(execution,candidate['layouts'])
        save_json(path,result)
        probe = windows[0][1][:128].cuda()[None]
        first = execution.backbone(probe,reset=True)
        if not bool(torch.isfinite(first).all()): raise CandidateInvalid('Nonfinite first128-token hidden')
        zero_scales = finite_cache(execution)
        cache0 = cache_identity(execution); hidden_sha = native.tensor_hash(first)
        second = execution.backbone(probe,reset=True)
        if not bool(torch.isfinite(second).all()): raise CandidateInvalid('Nonfinite repeated128-token hidden')
        finite_cache(execution)
        need(torch.equal(first,second) and cache0 == cache_identity(execution), 'Repeated reset output/cache differed')
        result['repeated_reset_probe'] = dict(tokens=128,hidden_sha256=hidden_sha,
            hidden_and_cache_exact=True,cache_tensor_sha256=cache0,
            token_sha256_int64le=runtime.token_digest(probe.cpu().numpy()),cache=require_cache(execution,128))
        result['storage_descriptor_probe'] = checked_descriptor(execution,candidate['layouts'])
        save_json(path,result)
        del first,second,probe,cache0
        total = 0.; count = 0
        for index,(start,window) in enumerate(windows):
            tokens = window.cuda(); hidden = execution.backbone(tokens[:-1][None],reset=True)
            if not bool(torch.isfinite(hidden).all()): raise CandidateInvalid('Nonfinite PPL hidden')
            zero_scales += finite_cache(execution)
            loss_sum = 0.
            for pos in range(0,hidden.shape[1],64):
                end = min(pos+64,hidden.shape[1])
                logits = model.lm_head(hidden[:,pos:end]).float()
                loss = F.cross_entropy(logits.reshape(-1,256000),tokens[pos+1:end+1],reduction='sum')
                loss_sum += float(loss); del logits,loss
            if not math.isfinite(loss_sum): raise CandidateInvalid('Nonfinite PPL NLL')
            targets = len(window)-1; total += loss_sum; count += targets
            result['ppl']['windows'].append(dict(start=start,target_tokens=targets,
                token_sha256_int64le=runtime.token_digest(window.numpy()),nll=loss_sum,ppl=finite_exp(loss_sum/targets)))
            result['ppl'].update(nll=total,target_tokens=count,ppl=finite_exp(total/count))
            result['cache'] = require_cache(execution,targets)
            result['storage_descriptor'] = checked_descriptor(execution,candidate['layouts'])
            result['ppl_end_cache_tensor_sha256'] = cache_identity(execution)
            save_json(path,result)
            if index == 0 or (index+1)%8 == 0 or index+1 == len(windows):
                print(f'[{common["arm"]} PPL] {index+1}/{len(windows)} ppl={result["ppl"]["ppl"]:.9f}',flush=True)
            del hidden,tokens
        result['ppl_cache'] = copy.deepcopy(result['cache'])
    result.update(complete=True,ppl_complete=True,runtime_table_unchanged=True,
        allocation_storage_validated=True,persistent_float_finite_checks_passed=True,
        controller_restoration=dict(complete=True,native_mixer_forwards_restored=True,native_hooks_restored=True),
        zero_scale_observations=zero_scales,gpu_memory=runtime.gpu_memory_receipt(),elapsed_seconds=time.time()-started)
    check_ppl(result,windows)
    return result


def valid_candidate(row):
    return (row.get('complete') is True and row.get('ppl_complete') is True and 'error' not in row
        and math.isfinite(row['ppl']['nll']) and math.isfinite(row['ppl']['ppl'])
        and row.get('cache',{}).get('total_bytes') == CACHE_BYTES
        and row.get('allocated_cache',{}).get('total_bytes') == CACHE_BYTES
        and row.get('persistent_float_finite_checks_passed') is True
        and row.get('repeated_reset_probe',{}).get('hidden_and_cache_exact') is True
        and row.get('runtime_table_unchanged') is True and row.get('candidate_table_unchanged') is True
        and row.get('allocation_storage_validated') is True
        and row.get('frozen_source',{}).get('identity_version_gradients_unchanged') is True
        and row.get('backend_policy_check',{}).get('singleton_config_unchanged') is True
        and row.get('controller_restoration',{}).get('native_mixer_forwards_restored') is True
        and row.get('controller_restoration',{}).get('native_hooks_restored') is True
        and row.get('adapter_hooks_absent') is True
        and row.get('adapter_loaded') is False and row.get('adapter_sha256') is None and row.get('mk_used') is False)


def choose_candidate(rows,order):
    need(order[0] == binding.BASELINE_ID and valid_candidate(rows[order[0]]), 'Valid fixed baseline required')
    valid = [name for name in order if valid_candidate(rows[name])]
    selected = min(valid,key=lambda name:(rows[name]['ppl']['nll'],name != binding.BASELINE_ID,order.index(name)))
    return dict(selected_id=selected,valid=valid,excluded=[name for name in order if name not in valid],
        baseline_id=binding.BASELINE_ID,baseline_wins=selected == binding.BASELINE_ID,
        rule='Minimum finite aggregate TRAIN NLL; exact ties baseline then fixed layout-major/table order',
        adapter_used=False,heldout_used=False,mk_used=False)


def exact_replay(before,after,*,archive=False):
    need(before.get('complete') is True and after.get('complete') is True, 'Replay requires complete finite PPL')
    for field in ('ppl','repeated_reset_probe','cache','ppl_cache','ppl_end_cache_tensor_sha256'):
        need(before[field] == after[field], 'Exact PPL/reset/cache replay differs: '+field)
    if archive:
        # V2 adds per-layer policy descriptions; archived physical buffers must still match exactly.
        physical = ('state_shape','state_bytes','tensors','conv_shape','conv_storage_bytes')
        for field in ('storage_descriptor_probe','storage_descriptor'):
            a,b = before[field]['layers'],after[field]['layers']
            need(len(a) == len(b) == 56 and all(all(left[k] == right[k] for k in physical)
                 for left,right in zip(a,b)), 'Archived physical storage descriptor differs')
    else:
        for field in ('allocated_cache','storage_descriptor_initial','storage_descriptor_probe','storage_descriptor'):
            need(before[field] == after[field], 'Restored physical allocation differs: '+field)
    return dict(complete=True,per_window_nll_exact=True,reset_output_and_cache_exact=True,
        physical_cache_exact=True,ppl_end_tensor_hashes_exact=True,
        windows=len(before['ppl']['windows']),target_tokens=before['ppl']['target_tokens'])


def load_tables(path):
    need(sha(path) == binding.RAW_STATS_SHA, 'Frozen original W4 calibration payload differs')
    raw = torch.load(path,map_location='cpu',weights_only=True)
    need(raw.get('format') == 'W4_STATE_CALIBRATION_RAW_V1' and raw.get('complete') is True
         and raw.get('w4_manifest_sha256') == base.W4_MANIFEST_SHA
         and raw.get('train_file_sha256') == base.TRAIN_SHA
         and raw.get('candidate_order') == list(binding.TABLE_ORDER)
         and all(raw.get(k) is False for k in ('adapter_used','heldout_used','mk_used')),
         'Frozen W4 tables must come from original unadapted TRAIN calibration')
    tables = raw['tables']
    need(set(tables) == set(binding.TABLE_ORDER), 'Fixed five-table inventory differs')
    for name,table in tables.items():
        need(base.check_table(table) == raw['table_sha256'][name], 'Frozen table hash differs: '+name)
    return tables,raw


def f1_candidates(tables):
    candidates = {}
    for layout in binding.LAYOUT_ORDER:
        for table_name in binding.TABLE_ORDER:
            name = layout+'__'+table_name
            candidates[name] = dict(id=name,layouts=(layout,)*56,table_name=table_name,table=tables[table_name])
    need(len(candidates) == 35 and next(iter(candidates)) == binding.BASELINE_ID, 'Fixed F1 grid/order differs')
    return candidates


def validate_kernel_inputs(checks_path,audit_path):
    checks = read_json(checks_path); audit = read_json(audit_path)
    need(checks.get('format') == 'W4_STATE_REPAIR_V2_CODEC_CHECK_V1'
         and checks.get('protocol_sha256') == binding.PROTOCOL_SHA
         and checks.get('complete') is True and checks.get('passed') is True
         and checks.get('cuda_initialized') is True and 'error' not in checks,
         'Passing actual GPU codec fixtures required before model-quality measurement')
    inventory = checks.get('code_sha256',checks.get('code_hashes'))
    check_source_inventory(inventory)
    need(inventory.get('scripts/w4_state_repair_codec_v2.py') == sha(ROOT/'scripts/w4_state_repair_codec_v2.py'),
         'GPU fixtures do not cover current repair codec')
    need(audit.get('format') == 'W4_STATE_REPAIR_V2_AUDIT_V1' and audit.get('stage') == 'codec_checks'
         and audit.get('complete') is True and audit.get('passed') is True
         and audit.get('cuda_initialized') is False and audit.get('protocol_sha256') == binding.PROTOCOL_SHA
         and audit.get('input_report_sha256') == sha(checks_path)
         and audit.get('source_sha256') == sha(ROOT/'scripts/audit_w4_state_repair_v2.py'),
         'Passing independent CPU codec audit with exact receipt/source binding required')
    return dict(kernel_checks_sha256=sha(checks_path),kernel_audit_sha256=sha(audit_path))


def export_selection(directory,comparison,candidates,common):
    name = comparison['selection']['selected_id']; candidate = candidates[name]
    payload = dict(format=SELECTION_FORMAT,complete=True,protocol_sha256=binding.PROTOCOL_SHA,
        w4_manifest_sha256=base.W4_MANIFEST_SHA,input_binding=common['input_binding'],
        selected_id=name,selected_layout=candidate['layouts'][0],selected_layer_layouts=list(candidate['layouts']),
        selected_table_name=candidate['table_name'],permutations=candidate['table'].clone().contiguous(),
        table_sha256=base.check_table(candidate['table']),scale_mode='stored_scale',int4_clip=1.,
        train_file_sha256=base.TRAIN_SHA,raw_stats_sha256=binding.RAW_STATS_SHA,
        screen_report_sha256=sha(directory/'screen_comparison.json'),
        kernel_checks_sha256=common['input_binding']['kernel_checks_sha256'],
        kernel_audit_sha256=common['input_binding']['kernel_audit_sha256'],code_sha256=code_hashes(),
        adapter_used=False,heldout_used=False,mk_used=False,cache_bytes=CACHE_BYTES,runtime_table_bytes=57344)
    path = directory/'selected_calibration.pt'; write_payload(path,payload)
    restored = torch.load(path,map_location='cpu',weights_only=True)
    need(torch.equal(restored['permutations'],payload['permutations']) and all(restored[k] == v
         for k,v in payload.items() if k != 'permutations'), 'Selected artifact serialization changed')
    receipt = {k:v for k,v in payload.items() if k != 'permutations'}
    receipt.update(file=path.name,sha256=sha(path),bytes=path.stat().st_size,serialization_roundtrip_bitwise=True)
    save_json(path.with_suffix('.json'),receipt)
    return receipt


def load_selection(path,audit_path,candidates,input_binding):
    payload = torch.load(path,map_location='cpu',weights_only=True); receipt = read_json(path.with_suffix('.json'))
    need(payload.get('format') == receipt.get('format') == SELECTION_FORMAT
         and payload.get('complete') is True and receipt.get('complete') is True
         and receipt.get('file') == path.name and receipt.get('sha256') == sha(path)
         and receipt.get('bytes') == path.stat().st_size, 'Complete frozen V2 selected artifact required')
    need(all(receipt.get(k) == v for k,v in payload.items() if k != 'permutations'),
         'Selected artifact and receipt metadata differ')
    need(payload.get('protocol_sha256') == binding.PROTOCOL_SHA
         and payload.get('input_binding') == input_binding and payload.get('code_sha256') == code_hashes()
         and payload.get('w4_manifest_sha256') == base.W4_MANIFEST_SHA
         and payload.get('train_file_sha256') == base.TRAIN_SHA
         and payload.get('raw_stats_sha256') == binding.RAW_STATS_SHA
         and payload.get('scale_mode') == 'stored_scale' and payload.get('int4_clip') == 1.
         and payload.get('cache_bytes') == CACHE_BYTES and payload.get('runtime_table_bytes') == 57344
         and all(payload.get(k) is False for k in ('adapter_used','heldout_used','mk_used')),
         'Selected artifact provenance differs')
    name = payload['selected_id']; need(name in candidates, 'Selected policy is outside fixed F1 grid')
    candidate = candidates[name]
    need(payload.get('selected_layout') == candidate['layouts'][0]
         and payload.get('selected_layer_layouts') == list(candidate['layouts'])
         and payload.get('selected_table_name') == candidate['table_name']
         and torch.equal(payload['permutations'],candidate['table'])
         and payload.get('table_sha256') == base.check_table(candidate['table']), 'Selected table/layout differs')
    comparison_path = path.parent/'screen_comparison.json'; comparison = read_json(comparison_path)
    need(payload['screen_report_sha256'] == sha(comparison_path) and comparison.get('format') == SCREEN_FORMAT
         and comparison.get('complete') is True and comparison.get('selection',{}).get('selected_id') == name
         and comparison.get('candidate_order') == list(candidates), 'Frozen TRAIN selection report differs')
    audit = read_json(audit_path)
    need(audit.get('format') == 'W4_STATE_REPAIR_V2_AUDIT_V1' and audit.get('stage') == 'screen'
         and audit.get('complete') is True and audit.get('passed') is True
         and audit.get('cuda_initialized') is False and audit.get('protocol_sha256') == binding.PROTOCOL_SHA
         and audit.get('input_report_sha256') == sha(comparison_path)
         and audit.get('source_sha256') == sha(ROOT/'scripts/audit_w4_state_repair_v2.py'),
         'Independent passing TRAIN screen audit required')
    return payload,receipt,candidate


def load_archive(path,windows):
    need(sha(path) == binding.BASELINE_REPORT_SHA, 'Original unadapted W4/SQ full PPL archive differs')
    archive = read_json(path)
    need(archive.get('complete') is True and archive.get('arm') == 'w4_sq325'
         and archive.get('w4_manifest_sha256') == base.W4_MANIFEST_SHA
         and archive.get('adapter_loaded') is False and archive.get('adapter_sha256') is None
         and archive.get('mk_used') is False and archive['ppl']['ppl'] == EXPECTED_BASELINE_PPL,
         'Archived W4/SQ baseline identity differs')
    check_ppl(archive,windows)
    return archive


def run_candidate(model,frozen,candidate,windows,path,common,policy,hashes,snapshot):
    before = base.check_table(candidate['table'])
    try:
        need(no_adapter_hooks(model), 'Unadapted repair evaluator found adapter hooks')
        result = evaluate(model,candidate,windows,path,common)
    except CandidateInvalid as error:
        result = read_json(path)
        result.update(complete=False,error=str(error),error_type=type(error).__name__,
            invalid_candidate=True,failure_kind='nonfinite_quality',fatal_failure=False)
    except BaseException as error:
        result = read_json(path) if path.exists() else dict(common,**candidate_metadata(candidate))
        result.update(complete=False,error=repr(error),error_type=type(error).__name__,
            failure_kind='integrity_or_runtime',fatal_failure=True)
        save_json(path,result); raise
    try:
        need(no_adapter_hooks(model), 'Adapter hooks appeared during unadapted repair')
        result['controller_restoration'] = check_native_snapshot(snapshot)
        result['frozen_source'] = frozen.check()
        result['backend_policy_check'] = check_replay_backend(policy)
        need(base.check_table(candidate['table']) == before and code_hashes() == hashes,
             'CPU table or execution source changed')
        result['candidate_table_unchanged'] = True
        result['adapter_hooks_absent'] = True
    except BaseException as error:
        result.update(complete=False,error=repr(error),error_type=type(error).__name__,
            failure_kind='integrity_or_runtime',fatal_failure=True)
        save_json(path,result); raise
    save_json(path,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('screen','full'),required=True)
    for name in ('w4-dir','source-dir','prose-tokens','kernel-checks','kernel-audit','out-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--raw-stats',type=Path,default=ROOT/'artifacts/calibration_v1/raw_stats.pt')
    parser.add_argument('--baseline-report',type=Path,default=ROOT/'artifacts/pretrain_v1/pretrain_w4_sq325.json')
    parser.add_argument('--selection','--calibration',dest='selection',type=Path)
    parser.add_argument('--screen-audit',type=Path)
    args = parser.parse_args()
    binding.check_protocol()
    need(not args.out_dir.exists() and not args.out_dir.is_symlink(), 'Fresh output directory required')
    kernel = validate_kernel_inputs(args.kernel_checks,args.kernel_audit)
    tables,raw = load_tables(args.raw_stats); candidates = f1_candidates(tables)
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256 == runtime.TOKENIZER_SHA256, 'Pinned tokenizer differs')
    train = load_train_tokens(args.prose_tokens)
    input_binding = dict(protocol_sha256=binding.PROTOCOL_SHA,w4_manifest_sha256=base.W4_MANIFEST_SHA,
        tokenizer_sha256=tokenizer.sha256,train_file_sha256=base.TRAIN_SHA,
        raw_stats_sha256=binding.RAW_STATS_SHA,baseline_report_sha256=binding.BASELINE_REPORT_SHA,**kernel)
    archive = selected = selected_receipt = selected_candidate = None
    if args.stage == 'screen':
        windows = [(row*2048,train[row].clone()) for row in SCREEN_ROWS]
        dataset = dict(split='train',file_sha256=base.TRAIN_SHA,rows=list(SCREEN_ROWS),
            tokens_per_row=2048,windows=32,target_tokens=65504)
        arms = [(name,candidate) for name,candidate in candidates.items()]
        arms.append((RESTORED,candidates[binding.BASELINE_ID]))
    else:
        need(args.selection is not None and args.screen_audit is not None, 'Full stage requires frozen selection and independent screen audit')
        selected,selected_receipt,selected_candidate = load_selection(args.selection,args.screen_audit,candidates,input_binding)
        need(selected['selected_id'] != binding.BASELINE_ID,
             'Unchanged parent won TRAIN; preserve the negative family and skip redundant full validation')
        ids,dataset = load_wikitext_tokens(tokenizer,'validation'); windows = ppl_windows(ids,2048)
        need(len(windows) == 130 and sum(len(w)-1 for _,w in windows) == 264764
             and dataset['token_stream_sha256_int64le'] == base.VALIDATION_TOKENS_SHA, 'Full validation population differs')
        archive = load_archive(args.baseline_report,windows)
        arms = [('baseline',candidates[binding.BASELINE_ID]),('selected',selected_candidate),
                (RESTORED,candidates[binding.BASELINE_ID])]
    torch.set_num_threads(8); torch.manual_seed(20260929); torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest'); policy = pin_replay_backend()
    if archive is not None: need(policy == archive['backend_policy'], 'Archived/current numerical backend differs')
    hashes = code_hashes(); args.out_dir.mkdir(parents=True)
    model = base.load_w4(args.w4_dir); frozen = FrozenBase(model); snapshot = native_snapshot(model)
    base_hashes = base.expected_base_hashes(args.w4_dir)
    common = dict(format=FORMAT,stage=args.stage,family='F1',protocol_sha256=binding.PROTOCOL_SHA,
        w4_manifest_sha256=base.W4_MANIFEST_SHA,input_binding=input_binding,dataset=dataset,
        tokenizer_sha256=tokenizer.sha256,train_file_sha256=base.TRAIN_SHA,
        raw_stats_sha256=binding.RAW_STATS_SHA,backend_policy=policy,code_sha256=hashes,
        w4_loader_receipt=model._package_receipt,environment=runtime.environment_receipt(),
        heldout_used=args.stage == 'full',heldout_used_for_selection=False,adapter_used=False,
        quality_scope='Historically exposed WikiText validation; no untouched-generalization claim')
    if selected is not None:
        common.update(selected_calibration_sha256=sha(args.selection),
            selected_calibration_receipt_sha256=sha(args.selection.with_suffix('.json')),
            screen_audit_sha256=sha(args.screen_audit),screen_report_sha256=selected['screen_report_sha256'])
    results = {}; started = time.time()
    for arm,candidate in arms:
        path = args.out_dir/(args.stage+'_'+arm+'.json')
        result = run_candidate(model,frozen,candidate,windows,path,dict(common,arm=arm),policy,hashes,snapshot)
        results[arm] = result
        if arm in (binding.BASELINE_ID,'baseline',RESTORED):
            need(valid_candidate(result), 'Baseline/restoration failed; stop and preserve numerical evidence')
        if archive is not None and arm in ('baseline',RESTORED):
            result['archived_ppl_replay'] = exact_replay(archive,result,archive=True)
            save_json(path,result)
    initial = binding.BASELINE_ID if args.stage == 'screen' else 'baseline'
    restoration = exact_replay(results[initial],results[RESTORED])
    final_base = base.assert_loaded_hashes(model,base_hashes)
    need(code_hashes() == hashes and no_adapter_hooks(model), 'Code/adapter integrity changed before final receipt')
    comparison = dict(format=SCREEN_FORMAT if args.stage == 'screen' else COMPARE_FORMAT,
        complete=True,stage=args.stage,family='F1',protocol_sha256=binding.PROTOCOL_SHA,
        w4_manifest_sha256=base.W4_MANIFEST_SHA,input_binding=input_binding,raw_stats_sha256=binding.RAW_STATS_SHA,
        dataset=dataset,backend_policy=policy,code_sha256=hashes,restoration=restoration,
        report_sha256={name:sha(args.out_dir/(args.stage+'_'+name+'.json')) for name in results},
        final_w4_content_check=final_base,adapter_used=False,mk_used=False,
        heldout_used=args.stage == 'full',heldout_used_for_selection=False,cache_bytes=CACHE_BYTES,
        all_integrity_checks_passed=True,elapsed_seconds=time.time()-started,
        independent_cpu_audit_required=True)
    if args.stage == 'screen':
        comparison.update(candidate_order=list(candidates),layout_order=list(binding.LAYOUT_ORDER),
            table_order=list(binding.TABLE_ORDER),selection=choose_candidate(results,list(candidates)))
        comparison.update(full_confirmation_required=not comparison['selection']['baseline_wins'],
            family_stopped=bool(comparison['selection']['baseline_wins']),
            target_repaired=False,
            outcome='unchanged_parent_won' if comparison['selection']['baseline_wins'] else 'train_winner_requires_full_confirmation')
        save_json(args.out_dir/'screen_comparison.json',comparison)
        receipt = export_selection(args.out_dir,comparison,candidates,common)
        print(json.dumps(dict(selected=receipt['selected_id'],table_sha256=receipt['table_sha256'],
            selected_calibration_sha256=receipt['sha256'],screen_report_sha256=receipt['screen_report_sha256']),indent=2),flush=True)
    else:
        candidate = results['selected']; finite = valid_candidate(candidate)
        target_checks = dict(ppl_strictly_below_8p4=finite and candidate['ppl']['ppl'] < binding.TARGET,
            cache_same_budget=all(row['allocated_cache']['total_bytes'] == CACHE_BYTES for row in results.values()),
            all_integrity_checks_passed=True,exact_archived_and_restored_replays=True,no_adapter=True)
        comparison.update(selected_id=selected['selected_id'],selected_layout=selected['selected_layout'],
            selected_layer_layouts=selected['selected_layer_layouts'],selected_table_name=selected['selected_table_name'],
            selected_table_sha256=selected['table_sha256'],selected_calibration_sha256=sha(args.selection),
            selected_calibration_receipt_sha256=sha(args.selection.with_suffix('.json')),
            screen_report_sha256=selected['screen_report_sha256'],screen_audit_sha256=sha(args.screen_audit),
            baseline_report_sha256=binding.BASELINE_REPORT_SHA,baseline_ppl=results['baseline']['ppl']['ppl'],
            selected_ppl=candidate['ppl']['ppl'] if finite else None,candidate_finite=finite,
            target= binding.TARGET,target_checks=target_checks,target_pass=all(target_checks.values()),
            archived_replays={name:results[name]['archived_ppl_replay'] for name in ('baseline',RESTORED)},
            outcome='finite_target_pass' if all(target_checks.values()) else 'finite_target_miss' if finite else 'nonfinite_quality',
            resurface_training_permitted=False,
            resurface_gate='New Resurface may be proposed only after strict target pass and independent full audit',
            memory=dict(row_bytes=52,ssm_bytes=23855104,conv_bytes=4587520,table_bytes=57344,
                cache_bytes=CACHE_BYTES,w4_file_bytes=4381415300,expanded_weight_payload_bytes=16473999360,
                adapter_bytes=0,scope='Persistent state, shared table, weight files, expanded parameters and measured GPU peaks are distinct'))
        save_json(args.out_dir/'full_comparison.json',comparison)
        print(json.dumps(comparison,indent=2),flush=True)


if __name__ == '__main__':
    main()
