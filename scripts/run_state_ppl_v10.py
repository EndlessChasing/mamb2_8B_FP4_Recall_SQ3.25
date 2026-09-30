#!/usr/bin/env python3
"""Disjoint TRAIN screening and frozen full PPL confirmation of same-byte global tier allocations."""
from __future__ import annotations

import argparse
import contextlib
import math
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime, resurface_data as data, resurface_native as native
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from prepare_quant_first import load_train_tokens, TRAIN_SHA, write_new_json
from prepare_state_first_v5 import need, read_json, tensor_sha, check_table, write_payload
from evaluate_quant_first import FrozenBase, VALIDATION_TOKENS_SHA
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from run_statequant import save_json
import run_state_ppl_v6 as v6
from run_state_ppl_v6 import CandidateInvalid, finite_cache, cache_identity, finite_exp, require_cache
import prepare_state_ppl_v9 as prep
import run_state_ppl_v9 as r9

PROTOCOL_SHA = '565707cf401b2ab472b9b78368a7192afeebe73056efd811b7acba610e3d9567'
LAYOUTS = {'16_64_48': (16,64,48), '8_80_40': (8,80,40), '24_48_56': (24,48,56), '32_32_64': (32,32,64)}
BASELINE = '16_64_48'
V9_SELECTED = '3467897358f33de22b1b629819cb2035f4cb8912914f0183959aa581fafeab50'
V9_SCREEN = '4098ad9446cb08d18c0ce23a8ce199c026a3836ae6fe5c5f3d21eadf2222d757'
V9_SCREEN_AUDIT = '4b218de5223ed35cf12be6c341e3dd3d5b75a82656019c8ac12526ce67f60402'
V9_TABLE = 'b1865e81ff3fbed028027a883872aef9bb91e614e7cf08c72211496a78aeb597'
SCREEN_ROWS = tuple(range(184,216))
NEW_INPUT_NAMES = ('group-candidates','v9-selected-calibration','v9-screen-audit','v9-full-dir','v9-full-audit','kernel-checks')
FORMAT = 'MAMBA2_STATE_PPL_V10_EVAL_V1'
COMPARE = 'MAMBA2_STATE_PPL_V10_COMPARISON_V1'
CALIBRATION = 'MAMBA2_STATE_PPL_V10_CALIBRATION_V1'
FULL_ARMS = ('v9_baseline', 'selected', 'restored_baseline')


def code_hashes():
    result = r9.code_hashes()
    for name in ('scripts/run_state_ppl_v10.py','scripts/state_ppl_codec_v10.py',
                 'scripts/check_state_ppl_codec_v10.py','scripts/audit_state_ppl_v10.py','docs/STATE_PPL_V10_PROTOCOL.md'):
        result[name] = data.sha_file(ROOT/name)
    return dict(sorted(result.items()))


def layout_spec(name):
    need(name in LAYOUTS,'Unknown fixed allocation')
    n8,n4,nzero = LAYOUTS[name]
    return dict(layout=name,n8=n8,n4=n4,nzero=nzero,row_bytes=52,payload_bytes=48,scale_bytes=4,
        resident_layout_metadata_bytes=0,tensor_widths=dict(lo=n8//2,hi=n8//2,q4=n4//2))


def candidate_spec(name, candidates):
    need(tuple(candidates['candidate_order']) == tuple(LAYOUTS) and name in LAYOUTS,'Fixed allocation grid differs')
    return dict(candidate_id=name,candidate_name='fixed_parent_allocation',variant='stored_scale',layout=name,
        scale_mode='stored_scale',int4_clip=1.,diagnostic=None,deployable=True)


def screen_arms(candidates):
    order = tuple(candidates['candidate_order'])
    need(order and order[0] == BASELINE and len(set(order)) == len(order)
         and 'restored_baseline' not in order, 'Invalid frozen allocation order')
    return (*order, 'restored_baseline')


def screen_dataset():
    return dict(split='train', file_sha256=TRAIN_SHA, rows=list(SCREEN_ROWS),
        tokens_per_row=2048, windows=32, target_tokens=65504)


def validate_storage_descriptor(actual, layout):
    descriptor=layout_spec(layout)
    need(all(actual.get(key)==value for key,value in descriptor.items()),'Allocation descriptor differs')
    need(len(actual['layers'])==56,'Actual allocation layer count differs')
    for layer in actual['layers']:
        need(layer['state_shape']==[1,128,64,128] and layer['state_bytes']==128*64*52
             and layer['conv_shape']==[1,10240,4] and layer['conv_storage_bytes']==10240*4*2,
             'Actual state/convolution geometry differs')
        need(set(layer['tensors'])=={'lo','hi','q4','s8','s4'},'Extra or missing resident state tensor')
        for name,width in descriptor['tensor_widths'].items():
            need(layer['tensors'][name]==dict(shape=[1,128,64,width],dtype='torch.uint8',storage_bytes=128*64*width),
                 'Actual packed tier tensor geometry differs: '+name)
        for name in ('s8','s4'):
            need(layer['tensors'][name]==dict(shape=[1,128,64],dtype='torch.float16',storage_bytes=128*64*2),
                 'Actual FP16 scale tensor geometry differs: '+name)
    return True


@contextlib.contextmanager
def guarded_execution(model,table,spec):
    from state_ppl_codec_v10 import StatePPLQuantV10
    with StatePPLQuantV10(model,table,layout=spec['layout']) as execution:
        digest=tensor_sha(table)
        try:yield execution
        finally:need(native.tensor_hash(execution.permutations)==digest,'Actual GPU permutation changed')


@torch.inference_mode()
def evaluate(model, table, spec, windows, path, common):
    result = dict(common, **spec, complete=False, ppl=dict(windows=[]))
    started = time.time()
    save_json(path, result)
    torch.cuda.reset_peak_memory_stats()
    with guarded_execution(model, table, spec) as execution:
        probe = windows[0][1][:128].cuda()[None]
        first = execution.backbone(probe, reset=True)
        if not bool(torch.isfinite(first).all()):
            raise CandidateInvalid('Nonfinite first128-token hidden')
        zero_scales = finite_cache(execution)
        cache0 = cache_identity(execution)
        hidden_sha = native.tensor_hash(first)
        second = execution.backbone(probe, reset=True)
        if not bool(torch.isfinite(second).all()):
            raise CandidateInvalid('Nonfinite repeated128-token hidden')
        finite_cache(execution)
        if not torch.equal(first, second) or cache0 != cache_identity(execution):
            raise RuntimeError('Repeated-reset128-token hidden/cache differed')
        result['repeated_reset_probe'] = dict(tokens=128, hidden_sha256=hidden_sha,
            hidden_and_cache_exact=True, cache_tensor_sha256=cache0,
            token_sha256_int64le=runtime.token_digest(probe.cpu().numpy()),
            cache=require_cache(execution, spec))
        result['storage_descriptor_probe'] = execution.storage_descriptor()
        validate_storage_descriptor(result['storage_descriptor_probe'],spec['layout'])
        del first, second, probe, cache0
        total = 0.
        count = 0
        for index, (start, window) in enumerate(windows):
            tokens = window.cuda()
            hidden = execution.backbone(tokens[:-1][None], reset=True)
            if not bool(torch.isfinite(hidden).all()):
                raise CandidateInvalid('Nonfinite PPL hidden')
            zero_scales += finite_cache(execution)
            loss_sum = 0.
            for pos in range(0, hidden.shape[1], 64):
                end = min(pos+64, hidden.shape[1])
                logits = model.lm_head(hidden[:,pos:end]).float()
                loss = F.cross_entropy(logits.reshape(-1,256000), tokens[pos+1:end+1], reduction='sum')
                loss_sum += float(loss)
                del logits, loss
            if not math.isfinite(loss_sum):
                raise CandidateInvalid('Nonfinite PPL loss')
            targets = len(window)-1
            total += loss_sum
            count += targets
            result['ppl']['windows'].append(dict(start=start, target_tokens=targets,
                token_sha256_int64le=runtime.token_digest(window.numpy()), nll=loss_sum,
                ppl=finite_exp(loss_sum/targets)))
            result['ppl'].update(nll=total, target_tokens=count, ppl=finite_exp(total/count))
            result['cache'] = require_cache(execution, spec)
            result['storage_descriptor'] = execution.storage_descriptor()
            validate_storage_descriptor(result['storage_descriptor'],spec['layout'])
            save_json(path, result)
            if index == 0 or (index+1) % 8 == 0 or index+1 == len(windows):
                print(f'[{common["arm"]} PPL] {index+1}/{len(windows)} ppl={result["ppl"]["ppl"]:.6f}', flush=True)
            del hidden, tokens
        result['persistent_float_finite_checks_passed'] = True
        result['zero_scale_observations'] = zero_scales
        result['zero_scale_scope'] = 'Final-cache observations of stored s4/s8; includes true zeros and scale underflow, not unique underflow events'
        result['gpu_memory'] = runtime.gpu_memory_receipt()
    result.update(complete=True, runtime_table_unchanged=True, allocation_storage_validated=True, elapsed_seconds=time.time()-started)
    return result


def validate_kernel(path):
    receipt=read_json(path)
    need(receipt.get('format')=='MAMBA2_STATE_PPL_V10_CODEC_CHECK_V1' and receipt.get('complete') is True
         and receipt.get('passed') is True and receipt.get('mode')=='gpu' and receipt.get('cuda_initialized') is True
         and 'error' not in receipt and receipt.get('protocol_sha256')==PROTOCOL_SHA,
         'Completed actual v10 GPU codec checks required')
    need(receipt.get('checks') and all(row.get('pass') is True for row in receipt['checks']), 'Every allocation codec check must pass')
    need({'scripts/state_ppl_codec_v10.py','scripts/check_state_ppl_codec_v10.py'} <= set(receipt['code_sha256']),
         'Incomplete allocation kernel source inventory')
    for name,digest in receipt['code_sha256'].items():
        need(data.sha_file(ROOT/name)==digest,'Codec check source differs: '+name)
    return receipt


def validate_parent_miss(args,parent):
    directory=args.v9_full_dir;path=directory/'full_comparison.json'
    comp=read_json(path);audit=read_json(args.v9_full_audit)
    dependencies={'audit_state_ppl_v8.py','audit_state_ppl_v6.py','audit_state_ppl_v7.py','audit_quant_first.py',
        'audit_state_repair.py','audit_state_first_v5.py','audit_resurface_more.py'}
    need(audit.get('format')=='MAMBA2_STATE_PPL_V9_INDEPENDENT_AUDIT_V1' and audit.get('complete') is True
         and audit.get('passed') is True and audit.get('cuda_initialized') is False and audit.get('stage')=='full'
         and audit.get('protocol_sha256')==prep.PROTOCOL_SHA
         and audit.get('source_sha256')==data.sha_file(ROOT/'scripts/audit_state_ppl_v9.py')
         and set(audit['auditor_dependency_sha256'])==dependencies and all(data.sha_file(ROOT/'scripts'/name)==digest
             for name,digest in audit['auditor_dependency_sha256'].items()),'Completed unchanged independent v9 full audit required')
    proof=audit['full']
    need(comp.get('format')==r9.COMPARE and comp.get('complete') is True and comp.get('stage')=='full'
         and comp.get('protocol_sha256')==prep.PROTOCOL_SHA and comp.get('code_hashes')==r9.code_hashes()
         and comp.get('adapter_loaded') is False and comp.get('adapter_sha256') is None and comp.get('mk_used') is False
         and comp.get('resurface_trained') is False and comp.get('selected_id')==parent['selected_id']=='top2'
         and comp.get('selected_calibration_sha256')==V9_SELECTED and comp.get('selection_report_sha256')==V9_SCREEN
         and audit['selected_calibration']['sha256']==V9_SELECTED and proof.get('complete') is True
         and proof['comparison_sha256']==data.sha_file(path) and proof['report_sha256']==comp['report_sha256']
         and tuple(comp['report_sha256'])==r9.FULL_ARMS,'v9 outcome does not bind frozen TRAIN top2')
    for arm,digest in comp['report_sha256'].items():
        need(data.sha_file(directory/('full_'+arm+'.json'))==digest,'Audited v9 full raw arm changed')
    checks=dict(ppl_strictly_below_8p25=False,cache_same_budget=True,all_integrity_checks_passed=True)
    need(comp.get('target_pass') is False and proof.get('target_pass') is False
         and comp['target_checks']==proof['target_checks']==checks,'v9 target passed or integrity failed; do not run v10 quality')
    archive=read_json(directory/'full_selected.json')
    need(v6.valid_candidate(archive) and archive['candidate_table_sha256']==V9_TABLE
         and math.isfinite(archive['ppl']['ppl']) and archive['ppl']['ppl']>=8.25,'Finite same-budget v9 miss required')
    return archive,dict(complete=True,comparison_sha256=data.sha_file(path),audit_sha256=data.sha_file(args.v9_full_audit),
        parent_report_sha256=data.sha_file(directory/'full_selected.json'),selected_calibration_sha256=V9_SELECTED,
        parent_ppl=archive['ppl']['ppl'],parent_selection='Frozen v9 TRAIN top2 irrespective of full comparison',
        criterion='Passing independent full audit; finite v9 PPL >=8.25; all integrity guards pass')


def load_inputs(args):
    need(data.sha_file(ROOT/'docs/STATE_PPL_V10_PROTOCOL.md')==PROTOCOL_SHA,'Frozen v10 protocol differs')
    validate_kernel(args.kernel_checks)
    old_candidates,old_binding,old_windows,_=r9.load_inputs(args)
    parent,_,_=r9.load_selection(args.v9_selected_calibration,old_candidates,old_binding,old_windows)
    need(data.sha_file(args.v9_selected_calibration)==V9_SELECTED and parent['selection_report_sha256']==V9_SCREEN
         and parent['table_sha256']==V9_TABLE and parent['selected_id']=='top2'
         and data.sha_file(args.v9_screen_audit)==V9_SCREEN_AUDIT,'Pinned v9 TRAIN parent differs')
    screen=read_json(args.v9_screen_audit)
    need(screen.get('complete') is True and screen.get('passed') is True and screen.get('stage')=='screen'
         and screen.get('cuda_initialized') is False and screen['screen']['comparison_sha256']==V9_SCREEN
         and screen['selected_calibration']['sha256']==V9_SELECTED,'Independent v9 screen audit differs')
    archive,start=validate_parent_miss(args,parent)
    candidates=dict(candidate_order=list(LAYOUTS),tables={name:parent['permutations'] for name in LAYOUTS},
        candidate_specs={name:layout_spec(name) for name in LAYOUTS})
    binding=dict(protocol_sha256=PROTOCOL_SHA,v9_input_binding=old_binding,
        v9_selected_calibration_sha256=V9_SELECTED,v9_selected_receipt_sha256=data.sha_file(args.v9_selected_calibration.with_suffix('.json')),
        v9_screen_audit_sha256=V9_SCREEN_AUDIT,v9_selection_report_sha256=V9_SCREEN,parent_table_sha256=V9_TABLE,
        start_condition=start,kernel_checks_sha256=data.sha_file(args.kernel_checks),
        source_sha256=runtime.SOURCE_CHECKPOINT_SHA256,tokenizer_sha256=runtime.TOKENIZER_SHA256,train_file_sha256=TRAIN_SHA)
    train=load_train_tokens(args.prose_tokens)
    windows=[(row*2048,train[row].clone()) for row in SCREEN_ROWS]
    need(len(windows)==32 and all(len(window)==2048 for _,window in windows)
         and sum(len(window)-1 for _,window in windows)==65504,'Fixed TRAIN population differs')
    return candidates,binding,windows,archive


def choose_candidate(rows, candidates):
    order = tuple(candidates['candidate_order'])
    need(v6.valid_candidate(rows[BASELINE]), 'Baseline must remain valid')
    valid = [name for name in order if v6.valid_candidate(rows[name])]
    selected = min(valid, key=lambda name: (rows[name]['ppl']['ppl'], name != BASELINE, order.index(name)))
    return dict(selected_id=selected, selected_variant='stored_scale', selected_layout=selected, baseline_id=BASELINE,
        valid=valid, excluded=[name for name in order if name not in valid], baseline_wins=selected == BASELINE,
        adapter_used=False, heldout_used=False, mk_used=False,
        rule='Minimum complete finite exact-budget disjoint TRAIN PPL; exact ties baseline then frozen candidate export order')


def export_selection(out, candidates, binding, comparison):
    selection = comparison['selection']
    name = selection['selected_id']
    table = candidates['tables'][name].clone().contiguous()
    payload = dict(format=CALIBRATION, protocol_sha256=PROTOCOL_SHA, input_binding=binding,
        **selection, selected_candidate_spec=candidates['candidate_specs'][name],
        permutations=table, table_sha256=tensor_sha(table), scale_mode='stored_scale', int4_clip=1.,
        runtime_table_bytes=57344, cache_bytes=v6.CACHE_BYTES, code_hashes=code_hashes(),
        selection_report_sha256=data.sha_file(out/'screen_comparison.json'))
    path = out/'selected_calibration.pt'
    write_payload(path, payload)
    receipt = {key: value for key, value in payload.items() if key != 'permutations'}
    receipt.update(complete=True, file=path.name, sha256=data.sha_file(path), bytes=path.stat().st_size)
    write_new_json(path.with_suffix('.json'), receipt)
    return receipt


def load_selection(path, candidates, binding, windows):
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    receipt = read_json(path.with_suffix('.json'))
    need(receipt.get('complete') is True and receipt.get('file') == path.name
         and receipt.get('sha256') == data.sha_file(path) and receipt.get('bytes') == path.stat().st_size
         and payload.get('format') == receipt.get('format') == CALIBRATION,
         'Selected allocation artifact/receipt differs')
    need(all(receipt.get(key) == value for key, value in payload.items() if key != 'permutations'),
         'Selected payload/receipt metadata differs')
    need(payload.get('protocol_sha256') == PROTOCOL_SHA and payload.get('input_binding') == binding
         and payload.get('code_hashes') == code_hashes() and payload.get('scale_mode') == 'stored_scale'
         and payload.get('int4_clip') == 1. and payload.get('runtime_table_bytes') == 57344
         and payload.get('cache_bytes') == v6.CACHE_BYTES, 'Selection source/input/storage policy differs')
    comp_path = path.parent/'screen_comparison.json'
    need(data.sha_file(comp_path) == payload['selection_report_sha256'], 'Selection comparison changed')
    comp = read_json(comp_path)
    need(comp.get('complete') is True and comp.get('format') == COMPARE and comp.get('stage') == 'screen'
         and comp.get('protocol_sha256') == PROTOCOL_SHA and comp.get('input_binding') == binding
         and comp.get('code_hashes') == code_hashes() and comp.get('adapter_loaded') is False
         and comp.get('adapter_sha256') is None and comp.get('heldout_used') is False
         and comp.get('heldout_used_for_selection') is False and comp.get('mk_used') is False,
         'Screen comparison provenance differs')
    arms = screen_arms(candidates)
    need(tuple(comp['report_sha256']) == arms, 'Screen arm population/order differs')
    identities = v6.window_identity(windows)
    rows = {}
    for arm in arms:
        arm_path = path.parent/('screen_'+arm+'.json')
        need(data.sha_file(arm_path) == comp['report_sha256'][arm], 'Screen report hash differs: '+arm)
        row = read_json(arm_path)
        name = BASELINE if arm == 'restored_baseline' else arm
        spec = candidate_spec(name, candidates)
        need(row.get('format') == FORMAT and row.get('stage') == 'screen' and row.get('arm') == arm
             and row.get('protocol_sha256') == PROTOCOL_SHA and row.get('input_binding') == binding
             and row.get('dataset') == screen_dataset() and row.get('code_hashes') == code_hashes()
             and row.get('candidate_table_sha256') == tensor_sha(candidates['tables'][name])
             and row.get('allocation_spec') == candidates['candidate_specs'][name]
             and row.get('adapter_loaded') is False and row.get('adapter_sha256') is None
             and row.get('heldout_used') is False and row.get('heldout_used_for_selection') is False
             and row.get('mk_used') is False and all(row.get(key) == value for key, value in spec.items())
             and all(row.get(key) is None for key in ('selected_calibration_sha256', 'parent_report_sha256',
                 'parent_comparison_sha256', 's16_report_sha256')), 'Screen row binding differs: '+arm)
        if row.get('complete'):
            v6.check_ppl(row, identities)
            need(v6.valid_candidate(row) and row.get('allocation_storage_validated') is True, 'Screen candidate integrity failed')
            validate_storage_descriptor(row['storage_descriptor_probe'],spec['layout'])
            validate_storage_descriptor(row['storage_descriptor'],spec['layout'])
        else:
            need(arm in candidates['candidate_order'] and arm != BASELINE
                 and row.get('error_type') == 'CandidateInvalid' and row.get('excluded_from_selection') is True
                 and not row.get('fatal_failure', False), 'Unrecognized incomplete screen arm')
        rows[arm] = row
    selection = choose_candidate(rows, candidates)
    need(comp['selection'] == selection and all(payload.get(key) == value for key, value in selection.items()),
         'Disjoint TRAIN selection rule differs')
    restoration = v6.check_restoration(rows[BASELINE], rows['restored_baseline'])
    need(comp['restoration'] == restoration and read_json(path.parent/'screen_restoration.json') == restoration,
         'Screen baseline restoration differs')
    need(comp['selected_vs_baseline'] == v6.compare_ppl(rows[BASELINE], rows[selection['selected_id']]),
         'Screen comparison arithmetic differs')
    table = payload['permutations']
    check_table(table)
    need(torch.equal(table, candidates['tables'][selection['selected_id']])
         and payload['table_sha256'] == tensor_sha(table)
         and payload['selected_candidate_spec'] == candidates['candidate_specs'][selection['selected_id']],
         'Selected allocation table/swaps differ')
    return payload, receipt, comp


def archive_replay(archived, current):
    need(archived.get('complete') is True and current.get('complete') is True
         and archived['ppl'] == current['ppl'], 'Archived v9 full NLL/PPL differs')
    need(archived['repeated_reset_probe'] == current['repeated_reset_probe'],
         'Archived v9 hidden/reset/cache probe differs')
    need(archived['cache'] == current['cache'] and current['cache']['total_bytes'] == v6.CACHE_BYTES,
         'Archived v9 actual cache allocation differs')
    return dict(complete=True, per_window_nll_exact=True, ppl_windows_repeated=130, target_tokens=264764,
        reset_hidden_exact=True, reset_cache_exact=True, cache_bytes_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen', 'full'), required=True)
    for name in (*prep.INPUT_NAMES,*NEW_INPUT_NAMES,'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('selected-calibration','s16-report'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    need(not args.out.exists(), 'Fresh output directory required; preserve prior evidence')
    candidates, binding, train_windows, parent_archive = load_inputs(args)
    start = binding['start_condition']
    selected = archive = s16 = None
    if args.stage == 'screen':
        need(args.selected_calibration is None and args.s16_report is None,
             'Screen accepts no candidate-validation or selected inputs')
        windows = train_windows
        dataset = screen_dataset()
        arms = [(arm, BASELINE if arm == 'restored_baseline' else arm) for arm in screen_arms(candidates)]
    else:
        need(args.selected_calibration is not None and args.s16_report is not None,
             'Full requires frozen selection and original S16 reference')
        selected, _, _ = load_selection(args.selected_calibration, candidates, binding, train_windows)
        need(not selected['baseline_wins'], 'Baseline winner does not advance to redundant full evaluation')
        need(data.sha_file(args.s16_report) == v6.S16_REPORT_SHA, 'Original S16 reference differs')
        archive = parent_archive
        s16 = read_json(args.s16_report)
        arms = [('v9_baseline', BASELINE), ('selected', selected['selected_id']), ('restored_baseline', BASELINE)]
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256 == runtime.TOKENIZER_SHA256, 'Tokenizer differs')
    if args.stage == 'full':
        ids, dataset = load_wikitext_tokens(tokenizer, 'validation')
        windows = ppl_windows(ids, 2048)
        need(len(windows) == 130 and sum(len(window)-1 for _, window in windows) == 264764
             and dataset['token_stream_sha256_int64le'] == VALIDATION_TOKENS_SHA, 'Validation population differs')
    torch.set_num_threads(8)
    torch.manual_seed(20260929); torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    policy = pin_replay_backend()
    args.out.mkdir(parents=True)
    model = runtime.load_source_model(args.source_dir)
    frozen = FrozenBase(model)
    need(v6.no_adapter_hooks(model), 'Unexpected source adapter hooks')
    common = dict(format=FORMAT, stage=args.stage, protocol_sha256=PROTOCOL_SHA, input_binding=binding,
        dataset=dataset, environment=runtime.environment_receipt(), code_hashes=code_hashes(), backend_policy=policy,
        adapter_loaded=False, adapter_sha256=None, mk_used=False, heldout_used=args.stage == 'full',
        heldout_used_for_selection=False,
        selected_calibration_sha256=data.sha_file(args.selected_calibration) if selected else None,
        parent_report_sha256=start['parent_report_sha256'] if archive else None,
        parent_comparison_sha256=start['comparison_sha256'] if archive else None,
        s16_report_sha256=v6.S16_REPORT_SHA if s16 else None)
    rows = {}
    started = time.time()
    identities = v6.window_identity(windows)
    parent_replay = None
    for arm, name in arms:
        print('[state-ppl-v10 arm] '+arm, flush=True)
        need(v6.no_adapter_hooks(model), 'Adapter hooks leaked')
        frozen.check()
        spec = candidate_spec(name, candidates)
        table = candidates['tables'][name]
        digest = tensor_sha(table)
        path = args.out/(args.stage+'_'+arm+'.json')
        arm_common = dict(common, arm=arm, candidate_table_sha256=digest,
            allocation_spec=candidates['candidate_specs'][name])
        try:
            result = evaluate(model, table, spec, windows, path, arm_common)
        except v6.CandidateInvalid as error:
            result = read_json(path)
            result.update(complete=False, error=repr(error), error_type='CandidateInvalid')
            if args.stage != 'screen' or name == BASELINE or arm == 'restored_baseline':
                result['fatal_failure'] = True; save_json(path, result)
                raise
            result['excluded_from_selection'] = True
        except Exception as error:
            result = read_json(path) if path.exists() else dict(arm_common, **spec)
            result.update(complete=False, error=repr(error), error_type=type(error).__name__, fatal_failure=True)
            save_json(path, result)
            raise
        result['frozen_source'] = frozen.check()
        result['backend_policy_check'] = check_replay_backend(policy)
        need(tensor_sha(table) == digest and v6.no_adapter_hooks(model), 'CPU table or adapter hooks changed')
        result['candidate_table_unchanged'] = True
        if result['complete']:
            v6.check_ppl(result, identities)
            need(v6.valid_candidate(result) and result.get('allocation_storage_validated') is True, 'Completed arm failed an integrity guard')
        save_json(path, result)
        rows[arm] = result
        if archive and arm == 'v9_baseline':
            parent_replay = archive_replay(archive, result)
            save_json(args.out/'full_parent_replay.json', parent_replay)
    baseline = BASELINE if args.stage == 'screen' else 'v9_baseline'
    restoration = v6.check_restoration(rows[baseline], rows['restored_baseline'])
    save_json(args.out/(args.stage+'_restoration.json'), restoration)
    comparison = dict(format=COMPARE, complete=True, stage=args.stage, protocol_sha256=PROTOCOL_SHA,
        input_binding=binding, code_hashes=code_hashes(), backend_policy=policy, restoration=restoration,
        adapter_loaded=False, adapter_sha256=None, mk_used=False, heldout_used=args.stage == 'full',
        heldout_used_for_selection=False,
        report_sha256={arm: data.sha_file(args.out/(args.stage+'_'+arm+'.json')) for arm in rows},
        elapsed_seconds=time.time()-started)
    if args.stage == 'screen':
        selection = choose_candidate(rows, candidates)
        comparison.update(selection=selection,
            selected_vs_baseline=v6.compare_ppl(rows[BASELINE], rows[selection['selected_id']]))
    else:
        checks = dict(ppl_strictly_below_8p25=rows['selected']['ppl']['ppl'] < 8.25,
            cache_same_budget=all(row['cache']['total_bytes'] == v6.CACHE_BYTES for row in rows.values()),
            all_integrity_checks_passed=all(v6.valid_candidate(row) and row.get('allocation_storage_validated') is True for row in rows.values()))
        comparison.update(selected_id=selected['selected_id'], selected_variant='stored_scale', selected_layout=selected['selected_layout'],
            selected_candidate_spec=selected['selected_candidate_spec'],
            selected_calibration_sha256=data.sha_file(args.selected_calibration),
            selection_report_sha256=selected['selection_report_sha256'], parent_report_sha256=start['parent_report_sha256'],
            parent_comparison_sha256=start['comparison_sha256'], s16_report_sha256=v6.S16_REPORT_SHA, parent_replay=parent_replay,
            comparison=v6.compare_ppl(rows['v9_baseline'], rows['selected']),
            original_s16_comparison=v6.compare_ppl(s16, rows['selected']),
            target_checks=checks, target_pass=all(checks.values()), resurface_trained=False, mk_evaluated=False)
    save_json(args.out/(args.stage+'_comparison.json'), comparison)
    if args.stage == 'screen':
        export_selection(args.out, candidates, binding, comparison)
        load_selection(args.out/'selected_calibration.pt', candidates, binding, train_windows)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
