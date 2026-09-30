#!/usr/bin/env python3
"""PPL-only TRAIN selection and frozen validation of same-budget state codecs."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime, resurface_data as data, resurface_native as native
from mamba2_recall.state_quant import StateQuant
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from prepare_quant_first import load_train_tokens, write_new_json, TRAIN_SHA, PROSE_MANIFEST_SHA
from prepare_state_first_v5 import (KINDS, CACHE_BYTES, PROTOCOL_SHA as V5_PROTOCOL_SHA,
    load_candidates, load_selected_calibration as load_v5_selection, check_table,
    need, read_json, tensor_sha, write_payload)
from evaluate_quant_first import FrozenBase, VALIDATION_TOKENS_SHA
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from run_statequant import save_json

PROTOCOL_SHA = '86d4e8dc85d79c2c867ce15a846c5938893c86d27e67ad174472975e654e9bff'
V5_CANDIDATES_SHA = 'cd86a755db5004c716922696cf5532b307c57f5fb7dad4bcca298eb58b55f9a7'
V5_SELECTED_SHA = 'c525fbf62ef4a72db2d4bb13c13920d9f5d4946485538da0f00a369aab3092e3'
V5_BASELINE_REPORT_SHA = '0f5412aac14801c7f5691df00308ea0c8ff1734f9d06bced276f38f871b58f69'
V5_COMPARISON_SHA = '0e5ceb6e91d72a159f46a9a0760a23f3b9301be18bb32590fe01a9196f83d1e9'
S16_REPORT_SHA = '52f82f83258a2fa3160ea14585f1bd69e1d68d60d8f179d1f0987c636f6546ba'
VARIANTS = ('legacy', 'stored_scale', 'clip4_095', 'clip4_090', 'clip4_080')
VARIANT_ARGUMENTS = dict(legacy=('legacy', 1.), stored_scale=('stored_scale', 1.),
    clip4_095=('stored_scale', .95), clip4_090=('stored_scale', .90),
    clip4_080=('stored_scale', .80))
CANDIDATE_IDS = tuple(kind+'__'+variant for kind in KINDS for variant in VARIANTS)
BASELINE_ID = 'preserve_int8__legacy'
DIAGNOSTIC_ARMS = ('diagnostic_s16', 'diagnostic_prune_only', 'diagnostic_quant_only')
SCREEN_ARMS = (BASELINE_ID, *DIAGNOSTIC_ARMS,
              *(name for name in CANDIDATE_IDS if name != BASELINE_ID), 'restored_baseline')
FULL_ARMS = ('v5_baseline', 'selected', 'restored_baseline')
EVAL_FORMAT = 'MAMBA2_STATE_PPL_EVAL_V1'
COMPARISON_FORMAT = 'MAMBA2_STATE_PPL_COMPARISON_V1'
CALIBRATION_FORMAT = 'MAMBA2_STATE_PPL_CALIBRATION_V1'


class CandidateInvalid(FloatingPointError):
    """Recognized nonfinite candidate result; integrity failures remain fatal."""


def check_protocol():
    need(data.sha_file(ROOT/'docs/STATE_PPL_V6_PROTOCOL.md') == PROTOCOL_SHA,
         'Frozen v6 protocol changed or is not yet bound')


def code_hashes():
    paths = list((ROOT/'mamba2_recall').glob('*.py'))
    paths += [ROOT/'scripts'/name for name in ('run_state_ppl_v6.py', 'state_ppl_codec_v6.py',
        'check_state_ppl_codec_v6.py', 'prepare_state_first_v5.py', 'prepare_quant_first.py',
        'evaluate_quant_first.py', 'evaluate_resurface_more.py', 'run_statequant.py')]
    paths += [ROOT/'docs/STATE_PPL_V6_PROTOCOL.md', ROOT/'docs/RESURFACE_MORE_BACKEND_REPLAY.md']
    return {str(path.relative_to(ROOT)): data.sha_file(path) for path in sorted(paths)}


def no_adapter_hooks(model):
    return not any(mx._forward_pre_hooks or mx._forward_hooks or mx.norm._forward_pre_hooks
                   for mx in (layer.mixer for layer in model.backbone.layers))


def finite_exp(value):
    try:
        result = math.exp(value)
    except OverflowError as error:
        raise CandidateInvalid('PPL exponent overflow') from error
    if not math.isfinite(result):
        raise CandidateInvalid('Nonfinite PPL')
    return result


def candidate_spec(candidate_id):
    need(candidate_id in CANDIDATE_IDS, 'Unknown fixed candidate')
    kind, variant = candidate_id.split('__')
    mode, clip = VARIANT_ARGUMENTS[variant]
    return dict(candidate_id=candidate_id, candidate_name=kind, variant=variant,
                scale_mode=mode, int4_clip=clip, diagnostic=None, deployable=True)


def diagnostic_spec(arm):
    need(arm in DIAGNOSTIC_ARMS, 'Unknown diagnostic')
    diagnostic = arm.removeprefix('diagnostic_')
    return dict(candidate_id=None, candidate_name=None if diagnostic == 's16' else 'preserve_int8',
        variant=None, scale_mode='legacy', int4_clip=1., diagnostic=diagnostic,
        deployable=False, selection_eligible=False,
        diagnostic_scope='TRAIN-only decomposition; excluded from candidate selection and Q3.25 claims')


def make_execution(model, table, spec):
    if spec['diagnostic'] == 's16':
        return StateQuant(model, 's16')
    from state_ppl_codec_v6 import StatePPLQuant
    return StatePPLQuant(model, table, scale_mode=spec['scale_mode'], int4_clip=spec['int4_clip'],
                         diagnostic=spec['diagnostic'])


@contextlib.contextmanager
def guarded_execution(model, table, spec):
    with make_execution(model, table, spec) as execution:
        digest = tensor_sha(table) if table is not None else None
        try:
            yield execution
        finally:
            need((execution.permutations is None if digest is None else
                  native.tensor_hash(execution.permutations) == digest),
                 'Actual GPU permutation table mutated during execution')


def expected_cache_bytes(spec):
    return 122028032 if spec['diagnostic'] == 's16' else 239525888 if spec['diagnostic'] else CACHE_BYTES


def cache_identity(execution):
    return {f'{index}.{key}': native.tensor_hash(value)
            for index, row in enumerate(execution._cache)
            for key, value in {**row.state.tensors, 'conv': row.conv}.items()}


def finite_cache(execution):
    floats = [value for row in execution._cache for value in row.state.tensors.values()
              if value.is_floating_point()]
    conv = [row.conv for row in execution._cache]
    if not bool(torch.stack([torch.isfinite(value).all() for value in floats+conv]).all()):
        raise CandidateInvalid('Nonfinite persisted state, FP16 scale, or convolution cache')
    scales = [value for row in execution._cache for name, value in row.state.tensors.items()
              if name in ('s4', 's8')]
    return int(torch.stack([(value == 0).sum() for value in scales]).sum()) if scales else 0


def require_cache(execution, spec):
    cache = execution.cache_breakdown()
    if (cache['total_bytes'] != expected_cache_bytes(spec) or cache['allocated_layers'] != 56
            or cache['batch_size'] != 1 or cache['conv_fp16_bytes'] != 4587520
            or cache['calibration_workspace_bytes'] != 0):
        raise RuntimeError('Actual persistent cache differs from fixed geometry/budget')
    if not spec['diagnostic'] and (cache['ssm_total_bytes'] != 23855104
            or cache['permutation_bytes'] != 57344):
        raise RuntimeError('Packed SSM or table allocation differs')
    return cache


def window_identity(windows):
    return [dict(start=start, target_tokens=len(window)-1,
        token_sha256_int64le=runtime.token_digest(window.numpy())) for start, window in windows]


def check_ppl(row, identities):
    need(row.get('complete') is True and 'error' not in row and 'mk' not in row,
         'Completed PPL-only arm required')
    actual = row['ppl']['windows']
    need(len(actual) == len(identities) and all(all(window.get(key) == value
        for key, value in identity.items()) for window, identity in zip(actual, identities)),
        'PPL window population differs')
    total = 0.
    count = 0
    for window in actual:
        loss, targets = window['nll'], window['target_tokens']
        need(math.isfinite(loss) and targets > 0 and loss >= 0
             and window['ppl'] == finite_exp(loss/targets), 'Invalid per-window NLL/PPL')
        total += loss
        count += targets
    need(row['ppl']['nll'] == total and row['ppl']['target_tokens'] == count
         and row['ppl']['ppl'] == finite_exp(total/count), 'Aggregate PPL arithmetic differs')


def valid_candidate(row):
    return (row.get('complete') is True and 'error' not in row and 'mk' not in row
        and row.get('deployable') is True and row.get('diagnostic') is None
        and math.isfinite(row['ppl']['ppl']) and row['cache']['total_bytes'] == CACHE_BYTES
        and row.get('persistent_float_finite_checks_passed') is True
        and row.get('repeated_reset_probe', {}).get('hidden_and_cache_exact') is True
        and row.get('frozen_source', {}).get('identity_version_gradients_unchanged') is True
        and row.get('backend_policy_check', {}).get('singleton_config_unchanged') is True
        and row.get('candidate_table_unchanged') is True
        and row.get('runtime_table_unchanged') is True
        and row.get('adapter_loaded') is False and row.get('adapter_sha256') is None)


def choose_candidate(results):
    need(valid_candidate(results[BASELINE_ID]), 'Pinned v5 baseline must be valid')
    valid = [name for name in CANDIDATE_IDS if valid_candidate(results[name])]
    selected = min(valid, key=lambda name: (results[name]['ppl']['ppl'],
        0 if name == BASELINE_ID else 1, CANDIDATE_IDS.index(name)))
    spec = candidate_spec(selected)
    return dict(selected_id=selected, selected_kind=spec['candidate_name'], selected_variant=spec['variant'],
        valid=valid, excluded=[name for name in CANDIDATE_IDS if name not in valid],
        stopped=selected == BASELINE_ID, adapter_used=False, heldout_used=False, mk_used=False,
        baseline_id=BASELINE_ID,
        rule='Fixed20 candidates; complete finite exact-budget; lowest TRAIN PPL, baseline priority, fixed table/variant order; diagnostics excluded')


def check_restoration(before, after):
    need(before.get('complete') is True and after.get('complete') is True,
         'Replay requires complete PPL arms')
    need(before['ppl']['windows'] == after['ppl']['windows'] and all(before['ppl'][key] == after['ppl'][key]
         for key in ('nll', 'target_tokens', 'ppl')), 'Exact per-window PPL replay failed')
    need(before['repeated_reset_probe'] == after['repeated_reset_probe'],
         'Exact reset hidden/cache replay failed')
    need(before['cache'] == after['cache'], 'Exact final persistent cache allocation replay failed')
    return dict(complete=True, per_window_nll_exact=True, ppl_windows_repeated=len(before['ppl']['windows']),
        target_tokens=before['ppl']['target_tokens'], reset_hidden_exact=True,
        reset_persistent_tensor_hashes_exact=True, cache_allocation_exact=True)


def check_archived_replay(archived, current):
    need(archived.get('complete') is True and current.get('complete') is True
         and archived['ppl']['windows'] == current['ppl']['windows']
         and all(archived['ppl'][key] == current['ppl'][key] for key in ('nll', 'target_tokens', 'ppl')),
         'Pinned v5 baseline per-window PPL replay failed')
    old_probe = archived['persistent_cache_probe']
    new_probe = current['repeated_reset_probe']
    need(old_probe['hidden_sha256'] == new_probe['hidden_sha256']
         and old_probe['cache_tensor_sha256'] == new_probe['cache_tensor_sha256']
         and archived['cache']['total_bytes'] == current['cache']['total_bytes'] == CACHE_BYTES,
         'Pinned v5 baseline hidden/cache probe replay failed')
    return dict(complete=True, per_window_nll_exact=True, ppl_windows_repeated=130,
        target_tokens=264764, reset_hidden_exact=True, reset_persistent_tensor_hashes_exact=True,
        cache_bytes_exact=True, mk_evaluated=False)


def compare_ppl(before, after):
    a, b = before['ppl']['windows'], after['ppl']['windows']
    need(len(a) == len(b) and all(all(left[key] == right[key]
        for key in ('start', 'target_tokens', 'token_sha256_int64le')) for left, right in zip(a,b)),
        'Compared PPL windows differ')
    return dict(control_arm=before['arm'], candidate_arm=after['arm'],
        control_ppl=before['ppl']['ppl'], candidate_ppl=after['ppl']['ppl'],
        ppl_relative_change=after['ppl']['ppl']/before['ppl']['ppl']-1,
        nll_delta=after['ppl']['nll']-before['ppl']['nll'],
        improved_windows=sum(y['nll'] < x['nll'] for x,y in zip(a,b)),
        regressed_windows=sum(y['nll'] > x['nll'] for x,y in zip(a,b)),
        unchanged_windows=sum(y['nll'] == x['nll'] for x,y in zip(a,b)))


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
            save_json(path, result)
            if index == 0 or (index+1) % 8 == 0 or index+1 == len(windows):
                print(f'[{common["arm"]} PPL] {index+1}/{len(windows)} ppl={result["ppl"]["ppl"]:.6f}', flush=True)
            del hidden, tokens
        result['persistent_float_finite_checks_passed'] = True
        result['zero_scale_observations'] = zero_scales
        result['zero_scale_scope'] = 'Final-cache observations of stored s4/s8; includes true zeros and scale underflow, not unique underflow events'
        result['gpu_memory'] = runtime.gpu_memory_receipt()
    result.update(complete=True, runtime_table_unchanged=True, elapsed_seconds=time.time()-started)
    return result


def validate_codec_checks(path):
    receipt = read_json(path)
    need(receipt.get('format') == 'MAMBA2_STATE_PPL_CODEC_CHECK_V1'
         and receipt.get('complete') is True and receipt.get('passed') is True
         and receipt.get('mode') == 'gpu' and receipt.get('cuda_initialized') is True
         and 'error' not in receipt, 'Completed actual GPU v6 codec checks required')
    need(receipt.get('checks') and all(row.get('pass') is True for row in receipt['checks']),
         'Every codec check must pass')
    need(receipt.get('protocol_sha256') == PROTOCOL_SHA, 'Codec-check protocol differs')
    need({'scripts/state_ppl_codec_v6.py', 'scripts/check_state_ppl_codec_v6.py'} <= set(receipt['code_sha256']),
         'Codec-check source inventory incomplete')
    for relative, digest in receipt['code_sha256'].items():
        need(data.sha_file(ROOT/relative) == digest, 'Codec check implementation changed: '+relative)
    return receipt


def load_inputs(candidates_path, v5_selected_path, codec_checks_path, prose_tokens_path):
    check_protocol()
    need(data.sha_file(candidates_path) == V5_CANDIDATES_SHA, 'Pinned v5 candidate artifact differs')
    need(data.sha_file(v5_selected_path) == V5_SELECTED_SHA, 'Pinned v5 selected calibration differs')
    candidates, candidate_receipt = load_candidates(candidates_path)
    v5_selected, _, _ = load_v5_selection(v5_selected_path, candidates_path)
    need(v5_selected['selected_kind'] == 'preserve_int8', 'v5 baseline table differs')
    checks = validate_codec_checks(codec_checks_path)
    train = load_train_tokens(prose_tokens_path)
    binding = dict(protocol_sha256=PROTOCOL_SHA, v5_protocol_sha256=V5_PROTOCOL_SHA,
        source_sha256=runtime.SOURCE_CHECKPOINT_SHA256, tokenizer_sha256=runtime.TOKENIZER_SHA256,
        train_file_sha256=TRAIN_SHA, prose_manifest_sha256=PROSE_MANIFEST_SHA,
        candidates_sha256=V5_CANDIDATES_SHA,
        candidates_receipt_sha256=data.sha_file(candidates_path.with_suffix('.json')),
        v5_selected_calibration_sha256=V5_SELECTED_SHA,
        v5_selected_receipt_sha256=data.sha_file(v5_selected_path.with_suffix('.json')),
        v5_selection_report_sha256=v5_selected['selection_report_sha256'],
        codec_checks_sha256=data.sha_file(codec_checks_path), table_sha256=candidate_receipt['table_sha256'],
        v4_statistics_sha256=v5_selected['v4_statistics_sha256'],
        original_calibration_sha256=v5_selected['original_calibration_sha256'])
    windows = [(index*2048, train[index,:2048]) for index in range(40,72)]
    need(len(windows) == 32 and sum(len(window)-1 for _,window in windows) == 65504,
         'Fixed v6 TRAIN screen population differs')
    return candidates, binding, windows, checks


def export_selection(out, candidates, binding, comparison):
    selection = comparison['selection']
    name = selection['selected_kind']
    common = dict(input_binding=binding, protocol_sha256=PROTOCOL_SHA,
        selected_id=selection['selected_id'], selected_kind=name,
        selected_variant=selection['selected_variant'], table_sha256=binding['table_sha256'][name],
        selection_report_sha256=data.sha_file(out/'screen_comparison.json'),
        stopped=selection['stopped'], adapter_loaded=False, adapter_sha256=None,
        heldout_used=False, mk_used=False)
    path = out/'selected_calibration.pt'
    payload = dict(format=CALIBRATION_FORMAT, **common, permutations=candidates['tables'][name].clone())
    write_payload(path, payload)
    receipt = dict(format=CALIBRATION_FORMAT, **common, complete=True,
        file=path.name, sha256=data.sha_file(path), bytes=path.stat().st_size, code_hashes=code_hashes())
    write_new_json(path.with_suffix('.json'), receipt)
    return receipt


def load_selected_calibration(path, candidates, binding, train_windows):
    """Validate immutable selected table/variant and independently replay TRAIN selection arithmetic."""
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    receipt = read_json(path.with_suffix('.json'))
    comparison_path = path.parent/'screen_comparison.json'
    comparison = read_json(comparison_path)
    need(payload.get('format') == receipt.get('format') == CALIBRATION_FORMAT
         and receipt.get('complete') is True and receipt.get('sha256') == data.sha_file(path)
         and receipt.get('bytes') == path.stat().st_size, 'Selected artifact/receipt identity differs')
    need(comparison.get('format') == COMPARISON_FORMAT and comparison.get('stage') == 'screen'
         and comparison.get('complete') is True and comparison.get('input_binding') == binding
         and comparison.get('protocol_sha256') == PROTOCOL_SHA
         and comparison.get('heldout_used') is False and comparison.get('adapter_loaded') is False
         and comparison.get('mk_used') is False and comparison.get('code_hashes') == code_hashes(),
         'Screen comparison provenance differs')
    need(tuple(comparison['report_sha256']) == SCREEN_ARMS, 'Screen arm sequence differs')
    results = {}
    identities = window_identity(train_windows)
    for arm in SCREEN_ARMS:
        arm_path = path.parent/('screen_'+arm+'.json')
        need(data.sha_file(arm_path) == comparison['report_sha256'][arm], 'Raw screen arm hash differs: '+arm)
        row = read_json(arm_path)
        spec = diagnostic_spec(arm) if arm in DIAGNOSTIC_ARMS else candidate_spec(BASELINE_ID if arm == 'restored_baseline' else arm)
        need(row.get('format') == EVAL_FORMAT and row.get('stage') == 'screen' and row.get('arm') == arm
             and row.get('input_binding') == binding and row.get('heldout_used') is False
             and row.get('adapter_loaded') is False and row.get('adapter_sha256') is None
             and row.get('mk_used') is False and row.get('code_hashes') == code_hashes()
             and all(row.get(key) == value for key,value in spec.items()), 'Screen arm binding differs: '+arm)
        if row.get('complete'):
            check_ppl(row, identities)
            need(row.get('persistent_float_finite_checks_passed') is True
                 and row.get('runtime_table_unchanged') is True and row.get('candidate_table_unchanged') is True
                 and row.get('frozen_source', {}).get('identity_version_gradients_unchanged') is True
                 and row.get('backend_policy_check', {}).get('singleton_config_unchanged') is True
                 and row.get('repeated_reset_probe', {}).get('hidden_and_cache_exact') is True
                 and row['cache']['total_bytes'] == expected_cache_bytes(spec),
                 'Completed screen arm integrity proof differs: '+arm)
        else:
            need(arm in CANDIDATE_IDS and arm != BASELINE_ID and row.get('excluded_from_selection') is True
                 and row.get('error_type') == 'CandidateInvalid', 'Unrecognized incomplete screening arm')
        results[arm] = row
    selection = choose_candidate(results)
    need(comparison['selection'] == selection, 'TRAIN-only selection arithmetic differs')
    need(comparison['restoration'] == check_restoration(results[BASELINE_ID], results['restored_baseline']),
         'Screen restoration proof differs')
    selected = selection['selected_id']
    name, variant = selected.split('__')
    expected = dict(input_binding=binding, protocol_sha256=PROTOCOL_SHA, selected_id=selected,
        selected_kind=name, selected_variant=variant, table_sha256=binding['table_sha256'][name],
        selection_report_sha256=data.sha_file(comparison_path), stopped=selection['stopped'],
        adapter_loaded=False, adapter_sha256=None, heldout_used=False, mk_used=False)
    for key,value in expected.items():
        need(payload.get(key) == receipt.get(key) == value, 'Selected binding differs: '+key)
    check_table(payload['permutations'])
    need(torch.equal(payload['permutations'], candidates['tables'][name])
         and tensor_sha(payload['permutations']) == expected['table_sha256'], 'Selected table differs')
    need(receipt.get('code_hashes') == code_hashes(), 'Selected implementation binding differs')
    return payload, receipt, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen','full'), required=True)
    for name in ('source-dir','candidates','v5-selected-calibration','prose-tokens','codec-checks','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--selected-calibration', type=Path)
    parser.add_argument('--parent-report', type=Path)
    parser.add_argument('--parent-comparison', type=Path)
    parser.add_argument('--s16-report', type=Path)
    args = parser.parse_args()
    need(not args.out.exists(), 'Fresh output directory required; preserve earlier evidence')
    candidates, binding, train_windows, _ = load_inputs(args.candidates, args.v5_selected_calibration,
        args.codec_checks, args.prose_tokens)
    selected = archive = comparison = original_s16 = None
    if args.stage == 'screen':
        need(all(value is None for value in (args.selected_calibration, args.parent_report,
             args.parent_comparison, args.s16_report)),
             'TRAIN screen cannot receive heldout or selected-candidate inputs')
        windows = train_windows
        dataset = dict(split='train', file_sha256=TRAIN_SHA, rows=list(range(40,72)),
            tokens_per_row=2048, windows=32, target_tokens=65504)
        arm_specs = [(arm, diagnostic_spec(arm) if arm in DIAGNOSTIC_ARMS
            else candidate_spec(BASELINE_ID if arm == 'restored_baseline' else arm)) for arm in SCREEN_ARMS]
    else:
        need(all(value is not None for value in (args.selected_calibration, args.parent_report,
             args.parent_comparison, args.s16_report)),
             'Full validation requires frozen TRAIN selection and pinned v5 baseline report')
        selected, _, comparison = load_selected_calibration(args.selected_calibration, candidates, binding, train_windows)
        need(not selected['stopped'], 'Baseline fallback stops; no redundant full evaluation')
        need(data.sha_file(args.parent_report) == V5_BASELINE_REPORT_SHA, 'Pinned v5 baseline report differs')
        need(data.sha_file(args.parent_comparison) == V5_COMPARISON_SHA, 'Pinned v5 comparison differs')
        historical = read_json(args.parent_comparison)
        need(historical['report_sha256']['selected_no_adapter'] == V5_BASELINE_REPORT_SHA
             and historical['archived_report_sha256']['source_s16'] == S16_REPORT_SHA,
             'Historical baseline/S16 report hashes do not bind the pinned comparison')
        need(data.sha_file(args.s16_report) == S16_REPORT_SHA, 'Pinned S16 context report differs')
        original_s16 = read_json(args.s16_report)
        archive = read_json(args.parent_report)
        arm_specs = [('v5_baseline', candidate_spec(BASELINE_ID)),
            ('selected', candidate_spec(selected['selected_id'])), ('restored_baseline', candidate_spec(BASELINE_ID))]
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256 == runtime.TOKENIZER_SHA256, 'Tokenizer differs')
    if args.stage == 'full':
        ids, dataset = load_wikitext_tokens(tokenizer, 'validation')
        windows = ppl_windows(ids,2048)
        need(len(windows) == 130 and sum(len(window)-1 for _,window in windows) == 264764
             and dataset['token_stream_sha256_int64le'] == VALIDATION_TOKENS_SHA, 'Validation population differs')
    torch.set_num_threads(8)
    torch.manual_seed(20260928)
    torch.cuda.manual_seed_all(20260928)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    policy = pin_replay_backend()
    args.out.mkdir(parents=True)
    model = runtime.load_source_model(args.source_dir)
    frozen = FrozenBase(model)
    need(no_adapter_hooks(model), 'Unexpected adapter hooks on source model')
    common = dict(format=EVAL_FORMAT, stage=args.stage, protocol_sha256=PROTOCOL_SHA, input_binding=binding,
        dataset=dataset, environment=runtime.environment_receipt(), code_hashes=code_hashes(), backend_policy=policy,
        adapter_loaded=False, adapter_sha256=None, mk_used=False,
        heldout_used=args.stage == 'full', heldout_used_for_selection=False,
        selection_scope='Unadapted PPL-only fixed TRAIN screen; full validation confirms a frozen table/variant',
        selected_calibration_sha256=data.sha_file(args.selected_calibration) if selected else None,
        parent_report_sha256=V5_BASELINE_REPORT_SHA if archive else None,
        parent_comparison_sha256=V5_COMPARISON_SHA if archive else None,
        s16_report_sha256=S16_REPORT_SHA if archive else None)
    results = {}
    started = time.time()
    parent_replay = None
    identities = window_identity(windows)
    for arm, spec in arm_specs:
        print('[state-ppl-v6 arm] '+arm, flush=True)
        need(no_adapter_hooks(model), 'Adapter hooks leaked into a PPL-only arm')
        frozen.check()
        table = candidates['tables'].get(spec['candidate_name'])
        table_digest = tensor_sha(table) if table is not None else None
        path = args.out/(args.stage+'_'+arm+'.json')
        arm_common = dict(common, arm=arm, candidate_table_sha256=table_digest)
        try:
            result = evaluate(model, table, spec, windows, path, arm_common)
        except CandidateInvalid as error:
            result = read_json(path)
            result.update(complete=False, error=repr(error), error_type='CandidateInvalid')
            if args.stage != 'screen' or arm not in CANDIDATE_IDS or arm == BASELINE_ID:
                result['fatal_failure'] = True
                save_json(path,result)
                raise
            result['excluded_from_selection'] = True
        except Exception as error:
            result = read_json(path) if path.exists() else dict(arm_common, **spec)
            result.update(complete=False, error=repr(error), error_type=type(error).__name__, fatal_failure=True)
            save_json(path,result)
            raise
        result['frozen_source'] = frozen.check()
        result['backend_policy_check'] = check_replay_backend(policy)
        need(table is None or tensor_sha(table) == table_digest, 'CPU candidate table mutated')
        result['candidate_table_unchanged'] = True
        need(no_adapter_hooks(model), 'Adapter hooks changed during evaluation')
        if result['complete']:
            check_ppl(result, identities)
        save_json(path,result)
        results[arm] = result
        if args.stage == 'full' and arm == 'v5_baseline':
            parent_replay = check_archived_replay(archive,result)
            save_json(args.out/'full_parent_replay.json',parent_replay)
    baseline_arm = BASELINE_ID if args.stage == 'screen' else 'v5_baseline'
    restoration = check_restoration(results[baseline_arm],results['restored_baseline'])
    save_json(args.out/(args.stage+'_restoration.json'),restoration)
    outcome = dict(format=COMPARISON_FORMAT, complete=True, stage=args.stage, protocol_sha256=PROTOCOL_SHA,
        input_binding=binding, code_hashes=code_hashes(), backend_policy=policy, restoration=restoration,
        adapter_loaded=False, adapter_sha256=None, mk_used=False, heldout_used=args.stage == 'full',
        heldout_used_for_selection=False,
        report_sha256={arm:data.sha_file(args.out/(args.stage+'_'+arm+'.json')) for arm in results},
        elapsed_seconds=time.time()-started)
    if args.stage == 'screen':
        selection = choose_candidate(results)
        diagnostics = {arm:compare_ppl(results['diagnostic_s16'],results[arm]) for arm in DIAGNOSTIC_ARMS[1:]}
        outcome.update(selection=selection, selected_table_sha256=binding['table_sha256'][selection['selected_kind']],
            diagnostics=diagnostics, diagnostics_used_for_selection=False,
            baseline_vs_s16=compare_ppl(results['diagnostic_s16'],results[BASELINE_ID]),
            selected_vs_baseline=compare_ppl(results[BASELINE_ID],results[selection['selected_id']]))
    else:
        difference = compare_ppl(results['v5_baseline'],results['selected'])
        checks = dict(ppl_at_least_1pct_better=results['selected']['ppl']['ppl'] <= .99*results['v5_baseline']['ppl']['ppl'],
            cache_same_budget=all(row['cache']['total_bytes'] == CACHE_BYTES for row in results.values()))
        outcome.update(selected_id=selected['selected_id'], selected_kind=selected['selected_kind'],
            selected_variant=selected['selected_variant'], selection=comparison['selection'],
            selected_calibration_sha256=data.sha_file(args.selected_calibration),
            selection_report_sha256=selected['selection_report_sha256'], parent_report_sha256=V5_BASELINE_REPORT_SHA,
            parent_comparison_sha256=V5_COMPARISON_SHA, s16_report_sha256=S16_REPORT_SHA,
            parent_replay=parent_replay, comparison=difference, gate_checks=checks,
            original_s16_comparison=compare_ppl(original_s16,results['selected']),
            gate_pass=all(checks.values()), resurface_trained=False,
            claim_scope='PPL and persistent-cache confirmation only; recall was not evaluated')
    save_json(args.out/(args.stage+'_comparison.json'),outcome)
    if args.stage == 'screen':
        receipt = export_selection(args.out,candidates,binding,outcome)
        load_selected_calibration(args.out/'selected_calibration.pt',candidates,binding,train_windows)
        print('[frozen selection] '+json.dumps({key:receipt[key] for key in
            ('selected_id','stopped','sha256','table_sha256')}),flush=True)
    print(json.dumps(outcome,indent=2),flush=True)


if __name__ == '__main__':
    main()
