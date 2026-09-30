#!/usr/bin/env python3
"""Evaluate the fixed final SQ3.25-first Resurface candidate and paired controls.

Pilot is descriptive, never a selection gate. Full evaluation is permitted
regardless of pilot quality and repeats the complete SQ baseline after adapter
removal. Only a completed formal 1536-update FP16 export is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mamba2_recall import resurface_data as data, resurface_native as native, runtime
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from run_statequant import evaluate, save_json

PROTOCOL_SHA = '24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb'
CALIBRATION_FORMAT = 'MAMBA2_QUANT_FIRST_CALIBRATION_V1'
TRAIN_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
PROSE_MANIFEST_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
VALIDATION_TOKENS_SHA = '5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027'
PILOT_WINDOWS = [0, 32, 64, 96]
PILOT_SAMPLES = [0, 9, 18, 27, 36, 45, 54, 63]
ARMS = ('source_s16', 'source_sq3p25', 'resurface_sq3p25', 'restored_source_sq3p25')


def code_hashes():
    paths = sorted((ROOT / 'mamba2_recall').glob('*.py'))
    paths += [Path(__file__), ROOT / 'scripts' / 'run_statequant.py',
              ROOT / 'docs' / 'QUANT_FIRST_PROTOCOL.md']
    return {str(path.relative_to(ROOT)): data.sha_file(path) for path in paths}


def load_candidate(args, tokenizer):
    """Bind source, no-adapter calibration, completed training, and real export."""
    if data.sha_file(ROOT / 'docs' / 'QUANT_FIRST_PROTOCOL.md') != PROTOCOL_SHA:
        raise ValueError('The frozen quantize-first protocol changed')
    calibration_sha = data.sha_file(args.calibration)
    calibration = torch.load(args.calibration, map_location='cpu', weights_only=True)
    receipt_path = args.calibration.with_suffix('.json')
    receipt = json.loads(receipt_path.read_text())
    expected_calibration = {
        'format': CALIBRATION_FORMAT, 'protocol_sha256': PROTOCOL_SHA,
        'source_sha256': runtime.SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256': tokenizer.sha256, 'train_file_sha256': TRAIN_SHA,
        'prose_manifest_sha256': PROSE_MANIFEST_SHA, 'adapter': None, 'adapter_sha256': None,
    }
    if (any(calibration.get(k) != v or receipt.get(k) != v for k, v in expected_calibration.items())
            or receipt.get('complete') is not True or receipt.get('fresh_source_no_adapter') is not True
            or receipt.get('sha256') != calibration_sha
            or receipt.get('bytes') != args.calibration.stat().st_size
            or receipt.get('heldout_used') is not False or receipt.get('calibration_tokens') != 4096
            or receipt.get('train_manifest_sha256') != calibration.get('train_manifest_sha256')
            or receipt.get('train_token_hashes') != calibration.get('train_token_hashes')):
        raise ValueError('Expected complete frozen calibration of the original unadapted source')
    permutations = calibration['permutations']
    stats = calibration['statistics']
    if (not isinstance(permutations, torch.Tensor) or permutations.dtype != torch.uint8
            or tuple(permutations.shape) != (56, 8, 128)
            or tuple(stats['mean_abs'].shape) != (56, 8, 128)
            or not bool(torch.isfinite(stats['mean_abs']).all())
            or not bool((stats['sample_count_per_group'] == 4096 * 16 * 64).all())
            or not torch.equal(permutations.sort(-1).values,
                               torch.arange(128, dtype=torch.uint8).expand_as(permutations))
            or not torch.equal(permutations, torch.argsort(stats['mean_abs'], dim=-1,
                                                          descending=True, stable=True).to(torch.uint8))
            or hashlib.sha256(permutations.numpy().tobytes()).hexdigest()
            != receipt.get('permutations_sha256_uint8')):
        raise ValueError('Frozen SQ3.25 tier tables/statistics are invalid')
    report = json.loads(args.training_report.read_text())
    if (report.get('format') != 'MAMBA2_SQ_FIRST_RESURFACE_TRAIN_V1'
            or report.get('complete') is not True or report.get('mode') != 'formal'
            or report.get('successful_updates') != 1536
            or not 1536 <= report.get('attempts', -1) <= 1544
            or report.get('initialization_check', {}).get('fresh_identity_matches_packed_bitwise') is not True
            or report.get('deployed_export_check', {}).get('packed_training_forward_bitwise_equal') is not True
            or report.get('frozen_base_check', {}).get('identity_version_gradients_unchanged') is not True
            or report.get('frozen_state_calibration_check') is not True
            or report.get('teacher_base_parameters_frozen') is not True):
        raise ValueError('A complete, verified formal 1536-update training report is required')
    history = report.get('history', [])
    if len(history) != report['attempts']:
        raise ValueError('Training attempt history is incomplete')
    schedule = torch.randperm(1536, generator=torch.Generator().manual_seed(2026092803)).tolist()
    successful = 0
    for index, row in enumerate(history, 1):
        if (successful >= 1536 or row.get('attempt') != index
                or row.get('schedule_entry') != schedule[successful]
                or type(row.get('overflow')) is not bool):
            raise ValueError('Training history differs from the frozen successful-update schedule')
        successful += int(not row['overflow'])
        if row.get('successful_updates') != successful:
            raise ValueError('Training successful-update accounting differs')
    if successful != 1536:
        raise ValueError('The final frozen candidate was not trained for1536 successful updates')
    binding = report['binding']
    required_binding = {
        'calibration_sha256': calibration_sha, 'calibration_format': CALIBRATION_FORMAT,
        'source_checkpoint_sha256': runtime.SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256': tokenizer.sha256, 'protocol_sha256': PROTOCOL_SHA,
        'train_manifest_sha256': calibration['train_manifest_sha256'],
        'prose_manifest_sha256': PROSE_MANIFEST_SHA, 'prose_tokens_sha256': TRAIN_SHA,
        'adapter': native.FORMAT, 'successful_updates': 1536,
        'initial_adapter': 'fresh V=0,g=1,w=0,b=-4; no pretrained adapter',
        'state_mode': 'sq3p25; exact packed forward; live-mask STE backward',
        'teacher': 'separate unadapted source, S16 per-token carry',
    }
    if any(binding.get(k) != v for k, v in required_binding.items()):
        raise ValueError('Final training binding differs from this source/calibration/protocol')
    deployed_files = ('runtime.py', 'state_codec.py', 'state_quant.py', 'resurface_native.py')
    for filename in deployed_files:
        relative = 'mamba2_recall/' + filename
        if report.get('code_sha256', {}).get(relative) != data.sha_file(ROOT / relative):
            raise ValueError('Deployment execution changed after fitting: ' + relative)
    exported = report['adapter']
    if Path(exported['file']).name != exported['file']:
        raise ValueError('Adapter filename must be relative to its training report directory')
    adapter_path = args.training_report.parent / exported['file']
    if (data.sha_file(adapter_path) != exported['sha256']
            or adapter_path.stat().st_size != exported['bytes']
            or exported.get('roundtrip_bitwise_equal') is not True
            or exported.get('parameters') != 1154104 or exported.get('gate_mode') != 'soft'):
        raise ValueError('The actual final FP16 export differs from the training receipt')
    adapter = native.read_fp16(adapter_path, expected_binding=binding)
    if (adapter['gate_mode'] != 'soft' or len(adapter['tensors']) != 224
            or sum(v.numel() for v in adapter['tensors'].values()) != 1154104
            or {k: native.tensor_hash(v) for k, v in adapter['tensors'].items()} != exported['tensor_sha256']):
        raise ValueError('Serialized adapter tensor identities/geometry differ')
    return permutations, receipt, report, adapter_path


class FrozenBase:
    """Cheap repeated checks of all original parameters; no full-byte hash claim."""
    def __init__(self, model):
        self.model = model
        params = dict(model.named_parameters())
        self.identities = {n: (id(p), p.data_ptr(), p._version) for n, p in params.items()}
        self.check()

    def check(self):
        params = dict(self.model.named_parameters())
        if set(params) != set(self.identities) or len(params) != 507:
            raise RuntimeError('Frozen source parameter inventory changed')
        if sum(p.numel() for p in params.values()) != 8236999680:
            raise RuntimeError('Wrong source parameter count')
        for name, value in params.items():
            if ((id(value), value.data_ptr(), value._version) != self.identities[name]
                    or value.dtype != torch.float16 or value.requires_grad or value.grad is not None):
                raise RuntimeError('Frozen source changed: ' + name)
        return {'tensors': 507, 'parameters': 8236999680,
                'identity_version_gradients_unchanged': True, 'actual_content_checked': False}


def compare_pair(control, candidate):
    if not control['complete'] or not candidate['complete']:
        raise ValueError('Comparison requires complete reports')
    if [(w['start'], w['target_tokens'], w['token_sha256_int64le']) for w in control['ppl']['windows']] != [
            (w['start'], w['target_tokens'], w['token_sha256_int64le']) for w in candidate['ppl']['windows']]:
        raise ValueError('PPL windows/token identities differ')
    left = {r['id']: r for r in control['mk']['rows'] if r['condition'] == 'normal'}
    right = {r['id']: r for r in candidate['mk']['rows'] if r['condition'] == 'normal'}
    if len(left) != control['mk']['summary']['normal']['count'] or set(left) != set(right) or not left:
        raise ValueError('Normal MK cases are not uniquely paired')
    for key in left:
        for field in ('prompt_token_sha256_int64le', 'answer', 'condition'):
            if left[key][field] != right[key][field]:
                raise ValueError('Paired MK case changed: ' + key)
    changes = np.asarray([int(right[k]['correct']) - int(left[k]['correct']) for k in sorted(left)])
    rng = np.random.default_rng(20260928)
    bootstrap = changes[rng.integers(0, len(changes), size=(10000, len(changes)))].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [.025, .975])
    p0, p1 = control['ppl']['ppl'], candidate['ppl']['ppl']
    return {'control_arm': control['arm'], 'candidate_arm': candidate['arm'],
            'control_ppl': p0, 'candidate_ppl': p1, 'ppl_relative_change': p1 / p0 - 1,
            'control_normal_mk': control['mk']['summary']['normal'],
            'candidate_normal_mk': candidate['mk']['summary']['normal'],
            'normal_mk_accuracy_delta': float(changes.mean()),
            'normal_mk_correct_delta': int(changes.sum()),
            'paired_improvements': int((changes > 0).sum()),
            'paired_regressions': int((changes < 0).sum()),
            'paired_unchanged': int((changes == 0).sum()),
            'normal_mk_paired_bootstrap_95ci': [float(lower), float(upper)],
            'bootstrap_draws': 10000, 'bootstrap_seed': 20260928,
            'control_target_removed_mk': control['mk']['summary']['target_removed'],
            'candidate_target_removed_mk': candidate['mk']['summary']['target_removed']}


def check_restoration(before, after):
    """Require all measured windows and complete greedy sequences to replay."""
    if not before['complete'] or not after['complete']:
        raise ValueError('Restoration requires both complete SQ evaluations')
    a, b = before['ppl']['windows'], after['ppl']['windows']
    if a != b:
        raise RuntimeError('Restored SQ baseline differs in a per-window NLL/token identity')
    if (before['ppl']['nll'], before['ppl']['ppl'], before['ppl']['target_tokens']) != (
            after['ppl']['nll'], after['ppl']['ppl'], after['ppl']['target_tokens']):
        raise RuntimeError('Restored aggregate PPL differs')
    x, y = before['mk']['rows'], after['mk']['rows']
    if len(x) != len(y) or not x:
        raise RuntimeError('Restored MK row inventory differs')
    for original, restored in zip(x, y):
        for field in ('id', 'prompt_token_sha256_int64le', 'generated_ids', 'output', 'prediction', 'correct'):
            if original[field] != restored[field]:
                raise RuntimeError('Restored SQ generation differs: ' + original['id'] + '/' + field)
    if before['cache']['total_bytes'] != after['cache']['total_bytes']:
        raise RuntimeError('Restored deployed cache allocation changed')
    return {'complete': True, 'per_window_nll_exact': True, 'all_generated_ids_exact': True,
            'all_decoded_predictions_exact': True, 'ppl_windows_repeated': len(a),
            'ppl_target_tokens_repeated': before['ppl']['target_tokens'], 'mk_cases_repeated': len(x),
            'scope': 'Entire selected pilot/full SQ baseline repeated after removing the new adapter'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--training-report', '--report', dest='training_report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('pilot', 'full'), default='pilot')
    args = parser.parse_args()
    if args.out.is_symlink():
        raise ValueError('Output symlinks are unsupported')
    names = [f'{args.split}_{arm}.json' for arm in ARMS]
    names += [f'{args.split}_comparison.json', f'{args.split}_restoration.json']
    if any((args.out / name).exists() or (args.out / name).is_symlink() for name in names):
        raise FileExistsError('Preserve existing evaluations; choose a fresh output path for this split')
    torch.set_num_threads(8)
    torch.manual_seed(20260928)
    torch.cuda.manual_seed_all(20260928)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    permutations, calibration, training, adapter_path = load_candidate(args, tokenizer)
    ids, dataset = load_wikitext_tokens(tokenizer, 'validation')
    windows = ppl_windows(ids, 2048)
    if (len(windows) != 130 or sum(len(w) - 1 for _, w in windows) != 264764
            or dataset['token_stream_sha256_int64le'] != VALIDATION_TOKENS_SHA):
        raise RuntimeError('Pinned validation token stream/window layout differs')
    if args.split == 'pilot':
        windows = [windows[i] for i in PILOT_WINDOWS]
        cases = [r for r in data.frozen_development_cases() if r['sample'] in PILOT_SAMPLES]
        expected_cases, expected_targets = 96, 8192
    else:
        cases = data.generate_cases('confirm')
        expected_cases, expected_targets = 768, 264764
    if (len(cases) != expected_cases or len({r['id'] for r in cases}) != expected_cases
            or sum(r['condition'] == 'normal' for r in cases) != expected_cases // 2
            or sum(len(w) - 1 for _, w in windows) != expected_targets):
        raise RuntimeError('Frozen evaluation case/window counts differ')
    args.out.mkdir(parents=True, exist_ok=True)
    model = runtime.load_source_model(args.source_dir)
    frozen = FrozenBase(model)
    common = {'format': 'MAMBA2_QUANT_FIRST_EVAL_V1', 'stage': args.split,
              'protocol_sha256': PROTOCOL_SHA, 'source_checkpoint_sha256': runtime.SOURCE_CHECKPOINT_SHA256,
              'tokenizer_sha256': tokenizer.sha256, 'calibration': calibration,
              'calibration_receipt_sha256': data.sha_file(args.calibration.with_suffix('.json')),
              'training_report_sha256': data.sha_file(args.training_report),
              'training_binding': training['binding'], 'candidate_adapter_sha256': training['adapter']['sha256'],
              'dataset': dataset, 'environment': runtime.environment_receipt(), 'code_hashes': code_hashes(),
              'execution': 'serial recurrence with per-token carried-state rounding/quantization; native prompt convolution and projections',
              'candidate_selection': 'Only final1536 successful updates; pilot never selects/rejects full evaluation',
              'quality_scope': 'Historically exposed validation corpus and DEV/CONFIRM prompt families; no unseen-generalization claim'}
    results = {}
    started = time.time()
    for arm in ARMS:
        mode = 's16' if arm == 'source_s16' else 'sq3p25'
        with_adapter = arm == 'resurface_sq3p25'
        bank = None
        print(f'[quant-first arm] {arm}', flush=True)
        try:
            if with_adapter:
                bank = native.install_fp16(model, adapter_path, expected_binding=training['binding'])
                adapter_hashes = {k: native.tensor_hash(v) for k, v in bank.masters.items()}
            elif any(mx._forward_pre_hooks or mx._forward_hooks or mx.norm._forward_pre_hooks
                     for mx in (layer.mixer for layer in model.backbone.layers)):
                raise RuntimeError('Baseline source unexpectedly retains adapter hooks')
            arm_common = {**common, 'arm': arm, 'adapter_sha256': training['adapter']['sha256'] if with_adapter else None}
            path = args.out / f'{args.split}_{arm}.json'
            result = evaluate(model, tokenizer, mode, permutations, windows, cases, path, arm_common)
            if result['ppl']['target_tokens'] != expected_targets or len(result['mk']['rows']) != expected_cases:
                raise RuntimeError('Completed evaluation omitted required targets/cases')
            result['frozen_source'] = frozen.check()
            if bank is not None:
                if adapter_hashes != {k: native.tensor_hash(v) for k, v in bank.masters.items()}:
                    raise RuntimeError('Frozen inference adapter values changed')
                result['adapter_content_unchanged'] = True
            save_json(path, result)
            results[arm] = result
        finally:
            if bank is not None:
                bank.close()
            frozen.check()
    restoration = check_restoration(results['source_sq3p25'], results['restored_source_sq3p25'])
    save_json(args.out / f'{args.split}_restoration.json', restoration)
    repair = compare_pair(results['source_sq3p25'], results['resurface_sq3p25'])
    source_gap = compare_pair(results['source_s16'], results['resurface_sq3p25'])
    quantization_effect = compare_pair(results['source_s16'], results['source_sq3p25'])
    point_gate = repair['ppl_relative_change'] <= .01 and repair['normal_mk_accuracy_delta'] > 0
    ci_supported = repair['normal_mk_paired_bootstrap_95ci'][0] > 0
    caches = {arm: result['cache']['total_bytes'] for arm, result in results.items()}
    if not caches['source_sq3p25'] == caches['resurface_sq3p25'] == caches['restored_source_sq3p25']:
        raise RuntimeError('Memoryless adapter unexpectedly changed deployed state-cache allocation')
    outcome = {'format': 'MAMBA2_QUANT_FIRST_COMPARISON_V1', 'complete': True, 'stage': args.split,
               'protocol_sha256': PROTOCOL_SHA, 'repair_vs_sq_baseline': repair,
               'remaining_gap_vs_original_s16': source_gap, 'sq_quantization_effect': quantization_effect,
               'ppl_no_worse_than_sq_baseline_1pct': repair['ppl_relative_change'] <= .01,
               'mk_observed_improvement_vs_sq': repair['normal_mk_accuracy_delta'] > 0,
               'mk_improvement_95ci_above_zero': ci_supported,
               'observed_joint_repair_gate_pass': bool(point_gate),
               'original_ppl_restored_within_1pct': source_gap['ppl_relative_change'] <= .01,
               'full_validation_complete': args.split == 'full',
               'final_claim_ready': bool(args.split == 'full' and point_gate and ci_supported),
               'status': ('Descriptive pilot only; run full regardless of these metrics' if args.split == 'pilot'
                          else 'Full paired repair gate passed with positive MK confidence interval' if point_gate and ci_supported
                          else 'Full observed repair gate passed; MK confidence interval includes zero' if point_gate
                          else 'Full paired repair gate not passed'),
               'restoration': restoration, 'cache_bytes': caches,
               'cache_reduction_fraction': 1 - caches['source_sq3p25'] / caches['source_s16'],
               'cache_scope': 'Batch1 persistent SSM and convolution cache plus compact tier tables; excludes weights/temporary workspace',
               'arm_seconds': {arm: result['elapsed_seconds'] for arm, result in results.items()},
               'elapsed_seconds': time.time() - started,
               'report_sha256': {arm: data.sha_file(args.out / f'{args.split}_{arm}.json') for arm in ARMS},
               'training_report_sha256': data.sha_file(args.training_report),
               'calibration_sha256': data.sha_file(args.calibration),
               'adapter_sha256': training['adapter']['sha256'], 'code_hashes': code_hashes(),
               'frozen_source_final': frozen.check()}
    save_json(args.out / f'{args.split}_comparison.json', outcome)
    print(json.dumps(outcome, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
