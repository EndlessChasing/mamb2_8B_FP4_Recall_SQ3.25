#!/usr/bin/env python3
"""Full paired evaluation of the fixed 4608-update SQ3.25 Resurface continuation."""
from __future__ import annotations
import argparse
import json
import inspect
import importlib.metadata
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mamba2_recall import resurface_data as data, resurface_native as native, runtime
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from mamba2_recall.state_quant import StateQuant
from run_statequant import evaluate, save_json
from evaluate_quant_first import (PROTOCOL_SHA as PARENT_PROTOCOL, VALIDATION_TOKENS_SHA,
    FrozenBase, load_candidate as load_parent, compare_pair, check_restoration)

PROTOCOL = '4c2c47aa00936ded52cf7b337126f1cce9556da7021e0a75c6e9df83f4949330'
PARENT_REPORT = 'e5a77d86cf2fb0e2389247e3cb325f74e89957861a6043e92a891d6d402ae359'
PARENT_CHECKPOINT = 'bc548dd427d114098048fa1863f8e602c095dc2d9fde56348795628ae8e2c78f'
PARENT_ADAPTER = '7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0'
CALIBRATION = 'c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023'
ARCHIVE_HASHES = {
    'source_s16': '52f82f83258a2fa3160ea14585f1bd69e1d68d60d8f179d1f0987c636f6546ba',
    'source_sq3p25': 'd8ec7b1ca239b385e444cfbec6c34f36728a459c6973c7f92690951e0447bc0d',
    'resurface_sq3p25': '224a8d1201761e44072bd37211ec0c916cc0872095042bf26cde49d8a349b20b',
}
ARMS = ('parent_resurface_sq3p25', 'continued_resurface_sq3p25', 'restored_parent_resurface_sq3p25')


def code_hashes():
    paths = sorted((ROOT/'mamba2_recall').glob('*.py'))
    paths += [Path(__file__), ROOT/'scripts/run_statequant.py', ROOT/'scripts/evaluate_quant_first.py',
              ROOT/'docs/RESURFACE_MORE_PROTOCOL.md', ROOT/'docs/QUANT_FIRST_PROTOCOL.md',
              ROOT/'docs/RESURFACE_MORE_BACKEND_REPLAY.md']
    return {str(p.relative_to(ROOT)): data.sha_file(p) for p in paths}



def _norm_config(config):
    return dict(kwargs=dict(config.kwargs), num_warps=config.num_warps,
                num_stages=config.num_stages, num_ctas=config.num_ctas,
                maxnreg=config.maxnreg, pre_hook_is_none=config.pre_hook is None)


def pin_replay_backend():
    """Select the archived RMSNorm reduction configuration before any forward.

    This selects by exact ORIGINAL/PARENT replay, never by candidate quality.
    Core files and the completed training remain unchanged.
    """
    from mamba_ssm.ops.triton import layer_norm
    from mamba_ssm.utils import determinism
    from triton.runtime import autotuner
    kernel = layer_norm._layer_norm_fwd_1pass_kernel
    original = [_norm_config(c) for c in kernel.configs]
    expected = [dict(kwargs={}, num_warps=w, num_stages=3, num_ctas=1,
                     maxnreg=None, pre_hook_is_none=True) for w in (1, 2, 4, 8, 16, 32)]
    if original != expected or kernel.cache:
        raise RuntimeError('Unexpected RMSNorm config inventory or a forward already selected a config')
    selected = next(c for c in kernel.configs if c.num_warps == 16)
    kernel.configs = [selected]
    kernel.cache.clear()
    flags = dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision())
    return dict(format='MAMBA2_ARCHIVED_RMSNORM_REPLAY_POLICY_V1',
        kernel='mamba_ssm.ops.triton.layer_norm._layer_norm_fwd_1pass_kernel',
        original_configs=original, selected_config=_norm_config(selected),
        singleton_config_bypasses_autotuning=True, applied_before_first_model_load=True,
        candidate_used_for_selection=False,
        selection_basis='Only warp16 exactly matches both archived S16 and parent first-window NLL; full archived parent replay remains mandatory',
        external_source_sha256={module.__name__:data.sha_file(Path(inspect.getfile(module)))
            for module in (layer_norm, determinism, autotuner)},
        package_versions={name:importlib.metadata.version(name) for name in ('torch','mamba-ssm','triton')},
        cudnn_version=torch.backends.cudnn.version(), precision_flags=flags,
        determinism_environment={key:os.environ.get(key) for key in ('MAMBA_DETERMINISTIC',
            'TRITON_CACHE_AUTOTUNING','TRITON_AUTOTUNE_BLOCK_SIZE_M','TRITON_AUTOTUNE_BLOCK_SIZE_N',
            'TRITON_AUTOTUNE_BLOCK_SIZE_K','TRITON_AUTOTUNE_BLOCK_SIZE_DSTATE')},
        clarification_sha256=data.sha_file(ROOT/'docs/RESURFACE_MORE_BACKEND_REPLAY.md'))


def check_replay_backend(policy):
    from mamba_ssm.ops.triton import layer_norm
    kernel = layer_norm._layer_norm_fwd_1pass_kernel
    if (len(kernel.configs) != 1 or _norm_config(kernel.configs[0]) != policy['selected_config']
            or (hasattr(kernel, 'best_config') and _norm_config(kernel.best_config) != policy['selected_config'])):
        raise RuntimeError('Pinned RMSNorm reduction configuration changed')
    return dict(singleton_config_unchanged=True,
                selected_config=_norm_config(kernel.configs[0]),
                best_config=_norm_config(kernel.best_config) if hasattr(kernel, 'best_config') else None)

def archive_reports(directory):
    results = {}
    for arm, digest in ARCHIVE_HASHES.items():
        path = directory/f'full_{arm}.json'
        if data.sha_file(path) != digest:
            raise ValueError('Archived v2 report digest differs: '+arm)
        value = json.loads(path.read_text())
        if (value.get('complete') is not True or value.get('stage') != 'full'
                or value.get('protocol_sha256') != PARENT_PROTOCOL
                or value.get('training_report_sha256') != PARENT_REPORT
                or value.get('candidate_adapter_sha256') != PARENT_ADAPTER):
            raise ValueError('Archived v2 report provenance differs: '+arm)
        results[arm] = value
    return results


def load_inputs(args, tokenizer):
    if (data.sha_file(ROOT/'docs/RESURFACE_MORE_PROTOCOL.md') != PROTOCOL
            or data.sha_file(args.parent_training_report) != PARENT_REPORT
            or data.sha_file(args.parent_checkpoint) != PARENT_CHECKPOINT
            or data.sha_file(args.calibration) != CALIBRATION):
        raise ValueError('Frozen continuation inputs changed')
    permutations, calibration, parent, parent_path = load_parent(SimpleNamespace(
        calibration=args.calibration, training_report=args.parent_training_report), tokenizer)
    if parent['adapter']['sha256'] != PARENT_ADAPTER:
        raise ValueError('Wrong parent adapter')
    report = json.loads(args.training_report.read_text())
    if (report.get('format') != 'MAMBA2_SQ_MORE_RESURFACE_TRAIN_V1'
            or report.get('complete') is not True or report.get('mode') != 'formal'
            or report.get('additional_successful_updates') != 3072
            or report.get('cumulative_successful_updates') != 4608
            or report.get('successful_updates') != 4608
            or not 3072 <= report.get('additional_attempts', -1) <= 3080
            or report.get('attempts') != report.get('additional_attempts') or 'error' in report
            or report.get('frozen_base_check', {}).get('identity_version_gradients_unchanged') is not True
            or report.get('frozen_state_calibration_check') is not True
            or report.get('teacher_base_parameters_frozen') is not True
            or report.get('deployed_export_check', {}).get('packed_training_forward_bitwise_equal') is not True):
        raise ValueError('A complete verified fixed 3072-additional/4608-total candidate is required')
    resume = report.get('parent_resume_check', {})
    if (any(resume.get(k) is not True for k in ('masters_exact', 'optimizer_exact', 'scaler_exact',
            'master_fp16_cast_equals_parent_export', 'packed_training_forward_bitwise_equal'))
            or resume.get('optimizer_steps') != 1536 or resume.get('optimizer_states') != 224
            or resume.get('master_tensors') != 224 or resume.get('probe_tokens') != 128
            or resume.get('scaler') != dict(scale=16., growth_factor=2., backoff_factor=.5,
                                           growth_interval=2000, _growth_tracker=1536)
            or resume.get('cache', {}).get('total_bytes') != 28499968):
        raise ValueError('Exact parent master/optimizer/scaler/forward restoration proof is missing')
    export_check = report.get('final_checkpoint_export_check', {})
    if (any(export_check.get(k) is not True for k in ('all_master_casts_equal_export', 'optimizer_exact', 'scaler_exact'))
            or export_check.get('master_tensors') != 224
            or report['deployed_export_check'].get('probe_tokens') != 128
            or report['deployed_export_check'].get('cache_unchanged_from_parent') is not True
            or report['deployed_export_check'].get('cache') != resume['cache']):
        raise ValueError('Final master/optimizer/scaler/export/cache proof missing')
    binding = report['binding']
    required = {**parent['binding'], 'continuation_protocol_sha256': PROTOCOL,
        'parent_training_report_sha256': PARENT_REPORT, 'parent_checkpoint_sha256': PARENT_CHECKPOINT,
        'parent_adapter_sha256': PARENT_ADAPTER, 'parent_successful_updates': 1536,
        'additional_successful_updates': 3072, 'successful_updates': 4608,
        'initial_adapter': 'exact parent FP32 masters/Adam/GradScaler checkpoint continuation'}
    if binding != required:
        raise ValueError('Continuation export binding differs')
    for relative, digest in report['code_sha256'].items():
        if data.sha_file(ROOT/relative) != digest:
            raise ValueError('Training/deployment code changed: '+relative)
    history = report.get('history', [])
    if len(history) != report['additional_attempts']:
        raise ValueError('Incomplete continuation attempt history')
    schedule = sum((torch.randperm(1536, generator=torch.Generator().manual_seed(seed)).tolist()
                    for seed in (2026092804, 2026092805)), [])
    successful = 0
    for index, row in enumerate(history, 1):
        if (successful >= 3072 or row['attempt'] != index or type(row['overflow']) is not bool
                or row['schedule_entry'] != schedule[successful]):
            raise ValueError('Continuation success/retry schedule changed')
        successful += int(not row['overflow'])
        if (row['additional_successful_updates'] != successful
                or row['cumulative_successful_updates'] != 1536+successful):
            raise ValueError('Continuation successful-update count differs')
    if successful != 3072:
        raise ValueError('Continuation final fixed candidate missing')
    exported = report['adapter']
    if Path(exported['file']).name != exported['file']:
        raise ValueError('Unsafe relative adapter path')
    adapter_path = args.training_report.parent/exported['file']
    if (data.sha_file(adapter_path) != exported['sha256'] or adapter_path.stat().st_size != exported['bytes']
            or exported['roundtrip_bitwise_equal'] is not True or exported['gate_mode'] != 'soft'
            or exported['parameters'] != 1154104):
        raise ValueError('Final adapter file differs')
    values = native.read_fp16(adapter_path, expected_binding=binding)['tensors']
    if len(values) != 224 or {k:native.tensor_hash(v) for k,v in values.items()} != exported['tensor_sha256']:
        raise ValueError('Final adapter tensors differ')
    if len(report['checkpoints']) != 4:
        raise ValueError('Four continuation recovery checkpoints required')
    final = report['checkpoints'][-1]
    path = args.training_report.parent/Path(final['path']).name
    if data.sha_file(path) != final['sha256'] or path.stat().st_size != final['bytes']:
        raise ValueError('Final checkpoint file differs')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'MAMBA2_SQ_MORE_RESURFACE_CHECKPOINT_V1'
            or checkpoint.get('additional_successful_updates') != 3072
            or checkpoint.get('cumulative_successful_updates') != 4608
            or checkpoint.get('additional_attempts') != report['additional_attempts']
            or checkpoint.get('attempts') != report['attempts']
            or export_check.get('final_checkpoint_sha256') != final['sha256']
            or checkpoint['scaler'] != report.get('final_scaler')
            or len(checkpoint['optimizer']['state']) != 224
            or any(s['step'].numel() != 1 or s['step'].item() != 4608
                   for s in checkpoint['optimizer']['state'].values())
            or checkpoint['binding'] != binding or checkpoint['successful_updates'] != 4608
            or set(checkpoint['masters']) != set(values)
            or any(v.dtype != torch.float32 or not torch.equal(v.half(), values[k])
                   for k,v in checkpoint['masters'].items())):
        raise ValueError('Actual final FP32 checkpoint does not cast exactly to the candidate export')
    return permutations, calibration, parent, parent_path, report, adapter_path


def replay_check(before, after, scope):
    result = check_restoration(before, after)
    result['scope'] = scope
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','calibration','parent-training-report','parent-checkpoint','training-report','parent-eval-dir','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    names = [f'full_{arm}.json' for arm in ARMS]+['full_comparison.json','full_restoration.json','full_parent_replay.json']
    if args.out.is_symlink() or any((args.out/n).exists() or (args.out/n).is_symlink() for n in names):
        raise FileExistsError('Preserve previous evidence; use a fresh evaluation output directory')
    torch.set_num_threads(8)
    torch.manual_seed(20260928); torch.cuda.manual_seed_all(20260928)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    permutations, calibration, parent, parent_path, training, adapter_path = load_inputs(args, tokenizer)
    archived = archive_reports(args.parent_eval_dir)
    ids, dataset = load_wikitext_tokens(tokenizer, 'validation')
    windows = ppl_windows(ids, 2048)
    cases = data.generate_cases('confirm')
    if (len(windows) != 130 or sum(len(w)-1 for _,w in windows) != 264764
            or dataset['token_stream_sha256_int64le'] != VALIDATION_TOKENS_SHA
            or len(cases) != 768 or len({c['id'] for c in cases}) != 768
            or sum(c['condition']=='normal' for c in cases) != 384):
        raise RuntimeError('Frozen full evaluation population differs')
    args.out.mkdir(parents=True, exist_ok=True)
    backend_policy = pin_replay_backend()
    model = runtime.load_source_model(args.source_dir)
    frozen = FrozenBase(model)
    common = dict(format='MAMBA2_MORE_RESURFACE_EVAL_V1', stage='full', protocol_sha256=PARENT_PROTOCOL,
        continuation_protocol_sha256=PROTOCOL, source_checkpoint_sha256=runtime.SOURCE_CHECKPOINT_SHA256,
        tokenizer_sha256=tokenizer.sha256, calibration=calibration,
        calibration_receipt_sha256=data.sha_file(args.calibration.with_suffix('.json')),
        parent_training_report_sha256=PARENT_REPORT, parent_checkpoint_sha256=PARENT_CHECKPOINT,
        parent_adapter_sha256=PARENT_ADAPTER, training_report_sha256=data.sha_file(args.training_report),
        training_binding=training['binding'], candidate_adapter_sha256=training['adapter']['sha256'],
        archived_report_sha256=ARCHIVE_HASHES, dataset=dataset,
        environment=runtime.environment_receipt(), code_hashes=code_hashes(), backend_policy=backend_policy,
        execution='serial recurrence with per-token carried-state rounding/quantization; native prompt convolution and projections',
        candidate_selection='Fixed final4608 successful updates; no DEV/CONFIRM checkpoint selection',
        quality_scope='Historically exposed validation corpus and CONFIRM families; no unseen-generalization claim')
    results = {}; started = time.time()
    for arm in ARMS:
        continued = arm == 'continued_resurface_sq3p25'
        report, path = (training, adapter_path) if continued else (parent, parent_path)
        bank = None
        print('[more-resurface arm] '+arm, flush=True)
        try:
            frozen.check()
            bank = native.install_fp16(model, path, expected_binding=report['binding'])
            hashes = {k:native.tensor_hash(v) for k,v in bank.masters.items()}
            if hashes != report['adapter']['tensor_sha256']:
                raise RuntimeError('Installed adapter differs from actual export')
            with StateQuant(model, 'sq3p25', permutations) as preflight:
                preflight.reset(1)
                cache_before = preflight.cache_breakdown()
            if cache_before['total_bytes'] != 28499968:
                raise RuntimeError('Actual pre-arm cache allocation changed')
            destination = args.out/f'full_{arm}.json'
            result = evaluate(model, tokenizer, 'sq3p25', permutations, windows, cases, destination,
                              {**common, 'arm':arm, 'adapter_sha256':report['adapter']['sha256']})
            if (len(result['ppl']['windows']) != 130 or result['ppl']['target_tokens'] != 264764
                    or len(result['mk']['rows']) != 768 or result['cache']['total_bytes'] != 28499968):
                raise RuntimeError('Full evaluation coverage/cache differs')
            with StateQuant(model, 'sq3p25', permutations) as postflight:
                postflight.reset(1)
                cache_after = postflight.cache_breakdown()
            if cache_after != cache_before:
                raise RuntimeError('Actual cache allocation differs after arm')
            result['cache_allocation_before'] = cache_before
            result['cache_allocation_after'] = cache_after
            result['frozen_source'] = frozen.check()
            if hashes != {k:native.tensor_hash(v) for k,v in bank.masters.items()}:
                raise RuntimeError('Inference changed the adapter contents')
            result['adapter_content_unchanged'] = True
            result['backend_policy_check'] = check_replay_backend(backend_policy)
            save_json(destination, result); results[arm] = result
            if arm == ARMS[0]:
                parent_replay = replay_check(archived['resurface_sq3p25'], result,
                    'Fresh parent exactly replays archived v2 full Resurface SQ3.25 NLL and generated IDs')
                save_json(args.out/'full_parent_replay.json', parent_replay)
        finally:
            if bank is not None:
                bank.close()
            frozen.check()
    restoration = replay_check(results[ARMS[0]], results[ARMS[2]],
        'Full parent repeated after removing the continued adapter and reinstalling the parent')
    save_json(args.out/'full_restoration.json', restoration)
    continuation = compare_pair(results[ARMS[0]], results[ARMS[1]])
    source_gap = compare_pair(archived['source_s16'], results[ARMS[1]])
    sq_repair = compare_pair(archived['source_sq3p25'], results[ARMS[1]])
    positive_ci = continuation['normal_mk_paired_bootstrap_95ci'][0] > 0
    point = continuation['ppl_relative_change'] <= .01 and continuation['normal_mk_accuracy_delta'] > 0
    outcome = dict(format='MAMBA2_MORE_RESURFACE_COMPARISON_V1', complete=True, stage='full',
        protocol_sha256=PARENT_PROTOCOL, continuation_protocol_sha256=PROTOCOL,
        continuation_vs_parent=continuation, repair_vs_archived_sq_baseline=sq_repair,
        remaining_gap_vs_archived_original_s16=source_gap,
        ppl_no_worse_than_parent_1pct=continuation['ppl_relative_change'] <= .01,
        ppl_improved_vs_parent=continuation['ppl_relative_change'] < 0,
        mk_observed_improvement_vs_parent=continuation['normal_mk_accuracy_delta'] > 0,
        mk_improvement_95ci_above_zero=positive_ci, observed_joint_gate_pass=bool(point),
        continuation_gate_pass=bool(point and positive_ci), final_claim_ready=bool(point and positive_ci),
        original_ppl_restored_within_1pct=source_gap['ppl_relative_change'] <= .01,
        full_validation_complete=True, parent_replay=parent_replay, restoration=restoration,
        cache_bytes={arm:results[arm]['cache']['total_bytes'] for arm in ARMS},
        cache_reduction_vs_archived_s16=1-28499968/archived['source_s16']['cache']['total_bytes'],
        cache_scope='Batch1 persistent SSM and convolution cache plus compact tier tables; excludes weights/temporary workspace',
        context_scope='S16 and unadapted SQ3.25 are hash-verified archived v2 context; all parent/continued arms are fresh full runs',
        status='Full continuation gate passed' if point and positive_ci else 'Full continuation gate not passed',
        arm_seconds={arm:results[arm]['elapsed_seconds'] for arm in ARMS}, elapsed_seconds=time.time()-started,
        report_sha256={arm:data.sha_file(args.out/f'full_{arm}.json') for arm in ARMS},
        archived_report_sha256=ARCHIVE_HASHES, training_report_sha256=data.sha_file(args.training_report),
        parent_training_report_sha256=PARENT_REPORT, parent_checkpoint_sha256=PARENT_CHECKPOINT,
        parent_adapter_sha256=PARENT_ADAPTER, calibration_sha256=CALIBRATION,
        adapter_sha256=training['adapter']['sha256'], code_hashes=code_hashes(), frozen_source_final=frozen.check(),
        backend_policy=backend_policy, backend_policy_check=check_replay_backend(backend_policy))
    save_json(args.out/'full_comparison.json', outcome)
    print(json.dumps(outcome, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
