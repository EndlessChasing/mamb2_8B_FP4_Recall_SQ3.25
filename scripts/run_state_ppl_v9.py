#!/usr/bin/env python3
"""Disjoint TRAIN screening and frozen full PPL confirmation of static group mixtures."""
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
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from prepare_quant_first import load_train_tokens, TRAIN_SHA, write_new_json
from prepare_state_first_v5 import need, read_json, tensor_sha, check_table, write_payload
from evaluate_quant_first import FrozenBase, VALIDATION_TOKENS_SHA
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from run_statequant import save_json
import run_state_ppl_v6 as v6
import prepare_state_ppl_v9 as prep

PROTOCOL_SHA = '9e01c03ee6870a8ecbcd9a0ba9157b1651d2830ea65d2951664ebcb81b4a09b3'
BASELINE = 'baseline'
FORMAT = 'MAMBA2_STATE_PPL_V9_EVAL_V1'
COMPARE = 'MAMBA2_STATE_PPL_V9_COMPARISON_V1'
CALIBRATION = 'MAMBA2_STATE_PPL_V9_CALIBRATION_V1'
FULL_ARMS = ('v8_baseline', 'selected', 'restored_baseline')


def code_hashes():
    result = prep.code_hashes()
    result['scripts/run_state_ppl_v9.py'] = data.sha_file(__file__)
    return dict(sorted(result.items()))


def candidate_spec(name, candidates):
    need(name in candidates['candidate_order'], 'Unknown frozen group-mixture candidate')
    spec = v6.candidate_spec(prep.p8.BASELINE_POLICY)
    spec.update(candidate_id=name, candidate_name='static_group_mix')
    return spec


def screen_arms(candidates):
    order = tuple(candidates['candidate_order'])
    need(order and order[0] == BASELINE and len(set(order)) == len(order)
         and 'restored_baseline' not in order, 'Invalid frozen group-mixture order')
    return (*order, 'restored_baseline')


def screen_dataset():
    return dict(split='train', file_sha256=TRAIN_SHA, rows=list(prep.SCREEN_ROWS),
        tokens_per_row=2048, windows=32, target_tokens=65504)


def load_inputs(args):
    need(prep.PROTOCOL_SHA == PROTOCOL_SHA, 'Preparation protocol differs')
    candidates, receipt, comparison, archive = prep.load_group_candidates(args.group_candidates,args)
    need(not candidates['calibration_stopped'] and len(candidates['candidate_order']) > 1,
         'No improving groups: protocol requires skipping redundant screen/full')
    train = load_train_tokens(args.prose_tokens)
    windows = [(row*2048, train[row].clone()) for row in prep.SCREEN_ROWS]
    need(len(windows) == 32 and all(len(window) == 2048 for _, window in windows)
         and sum(len(window)-1 for _, window in windows) == 65504, 'Disjoint TRAIN screen differs')
    binding = dict(protocol_sha256=PROTOCOL_SHA,
        calibration_input_binding=candidates['input_binding'],
        group_candidates_sha256=receipt['sha256'],
        group_candidates_receipt_sha256=data.sha_file(args.group_candidates.with_suffix('.json')),
        group_calibration_report_sha256=candidates['calibration_report_sha256'],
        parent_policy='Frozen v8 TRAIN top8 with v6 stored_scale',
        source_sha256=runtime.SOURCE_CHECKPOINT_SHA256,
        tokenizer_sha256=runtime.TOKENIZER_SHA256, train_file_sha256=TRAIN_SHA)
    screen_arms(candidates)
    need(comparison['ranking']['stopped'] == candidates['calibration_stopped'],
         'Calibration stopped flag differs')
    return candidates, binding, windows, archive


def choose_candidate(rows, candidates):
    order = tuple(candidates['candidate_order'])
    need(v6.valid_candidate(rows[BASELINE]), 'Baseline must remain valid')
    valid = [name for name in order if v6.valid_candidate(rows[name])]
    selected = min(valid, key=lambda name: (rows[name]['ppl']['ppl'], name != BASELINE, order.index(name)))
    return dict(selected_id=selected, selected_variant='stored_scale', baseline_id=BASELINE,
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
         'Selected group-mixture artifact/receipt differs')
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
             and row.get('group_mix_spec') == candidates['candidate_specs'][name]
             and row.get('adapter_loaded') is False and row.get('adapter_sha256') is None
             and row.get('heldout_used') is False and row.get('heldout_used_for_selection') is False
             and row.get('mk_used') is False and all(row.get(key) == value for key, value in spec.items())
             and all(row.get(key) is None for key in ('selected_calibration_sha256', 'parent_report_sha256',
                 'parent_comparison_sha256', 's16_report_sha256')), 'Screen row binding differs: '+arm)
        if row.get('complete'):
            v6.check_ppl(row, identities)
            need(v6.valid_candidate(row), 'Screen candidate integrity failed')
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
         'Selected group-mixture table/swaps differ')
    return payload, receipt, comp


def archive_replay(archived, current):
    need(archived.get('complete') is True and current.get('complete') is True
         and archived['ppl'] == current['ppl'], 'Archived v8 full NLL/PPL differs')
    need(archived['repeated_reset_probe'] == current['repeated_reset_probe'],
         'Archived v8 hidden/reset/cache probe differs')
    need(archived['cache'] == current['cache'] and current['cache']['total_bytes'] == v6.CACHE_BYTES,
         'Archived v8 actual cache allocation differs')
    return dict(complete=True, per_window_nll_exact=True, ppl_windows_repeated=130, target_tokens=264764,
        reset_hidden_exact=True, reset_cache_exact=True, cache_bytes_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen', 'full'), required=True)
    for name in (*prep.INPUT_NAMES,'group-candidates','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('selected-calibration','s16-report'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    need(not args.out.exists(), 'Fresh output directory required; preserve prior evidence')
    candidates, binding, train_windows, parent_archive = load_inputs(args)
    start = candidates['input_binding']['start_condition']
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
        arms = [('v8_baseline', BASELINE), ('selected', selected['selected_id']), ('restored_baseline', BASELINE)]
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
        print('[state-ppl-v9 arm] '+arm, flush=True)
        need(v6.no_adapter_hooks(model), 'Adapter hooks leaked')
        frozen.check()
        spec = candidate_spec(name, candidates)
        table = candidates['tables'][name]
        digest = tensor_sha(table)
        path = args.out/(args.stage+'_'+arm+'.json')
        arm_common = dict(common, arm=arm, candidate_table_sha256=digest,
            group_mix_spec=candidates['candidate_specs'][name])
        try:
            result = v6.evaluate(model, table, spec, windows, path, arm_common)
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
            need(v6.valid_candidate(result), 'Completed arm failed an integrity guard')
        save_json(path, result)
        rows[arm] = result
        if archive and arm == 'v8_baseline':
            parent_replay = archive_replay(archive, result)
            save_json(args.out/'full_parent_replay.json', parent_replay)
    baseline = BASELINE if args.stage == 'screen' else 'v8_baseline'
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
            all_integrity_checks_passed=all(v6.valid_candidate(row) for row in rows.values()))
        comparison.update(selected_id=selected['selected_id'], selected_variant='stored_scale',
            selected_candidate_spec=selected['selected_candidate_spec'],
            selected_calibration_sha256=data.sha_file(args.selected_calibration),
            selection_report_sha256=selected['selection_report_sha256'], parent_report_sha256=start['parent_report_sha256'],
            parent_comparison_sha256=start['comparison_sha256'], s16_report_sha256=v6.S16_REPORT_SHA, parent_replay=parent_replay,
            comparison=v6.compare_ppl(rows['v8_baseline'], rows['selected']),
            original_s16_comparison=v6.compare_ppl(s16, rows['selected']),
            target_checks=checks, target_pass=all(checks.values()), resurface_trained=False, mk_evaluated=False)
    save_json(args.out/(args.stage+'_comparison.json'), comparison)
    if args.stage == 'screen':
        export_selection(args.out, candidates, binding, comparison)
        load_selection(args.out/'selected_calibration.pt', candidates, binding, train_windows)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
