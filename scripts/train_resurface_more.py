#!/usr/bin/env python3
"""Continue SQ3.25 Resurface from its exact v2 FP32/Adam/scaler checkpoint.

Only the final 3072-additional/4608-cumulative update export is a candidate.
The one-update --smoke run is discarded and the formal run reloads the parent.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall import resurface_data as data, resurface_native as native
from mamba2_recall import resurface_loss as objective, runtime
from mamba2_recall.state_quant import StateQuant
from mamba2_recall.state_training import StateQuantTraining
from train_quant_first import load_training_inputs, optimizer_for, write_json
from evaluate_quant_first import load_candidate as load_parent_candidate

PARENT_STEPS = 1536
STEPS = 3072
PROTOCOL_SHA = '4c2c47aa00936ded52cf7b337126f1cce9556da7021e0a75c6e9df83f4949330'
PARENT_PROTOCOL_SHA = '24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb'
PARENT_REPORT_SHA = 'e5a77d86cf2fb0e2389247e3cb325f74e89957861a6043e92a891d6d402ae359'
PARENT_CHECKPOINT_SHA = 'bc548dd427d114098048fa1863f8e602c095dc2d9fde56348795628ae8e2c78f'
PARENT_ADAPTER_SHA = '7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0'
CALIBRATION_SHA = 'c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023'
FORMAT = 'MAMBA2_SQ_MORE_RESURFACE_TRAIN_V1'
CHECKPOINT_FORMAT = 'MAMBA2_SQ_MORE_RESURFACE_CHECKPOINT_V1'
INITIAL_ADAPTER = 'exact parent FP32 masters/Adam/GradScaler checkpoint continuation'
EXPECTED_SCALER = {'scale': 16.0, 'growth_factor': 2.0, 'backoff_factor': 0.5,
                   'growth_interval': 2000, '_growth_tracker': 1536}
CACHE_BYTES = 28499968


def lr_factor(r):
    if not 0 <= r < STEPS:
        raise ValueError('Additional successful update index outside frozen tail')
    return .01 + .09 * (1 + math.cos(math.pi * r / (STEPS - 1))) / 2


def schedule():
    return sum((torch.randperm(PARENT_STEPS,
        generator=torch.Generator(device='cpu').manual_seed(seed)).tolist()
        for seed in (2026092804, 2026092805)), [])


def code_hashes():
    paths = sorted((ROOT/'mamba2_recall').glob('*.py'))
    paths += [Path(__file__), ROOT/'scripts'/'train_quant_first.py',
              ROOT/'scripts'/'evaluate_quant_first.py', ROOT/'scripts'/'run_statequant.py',
              ROOT/'docs'/'QUANT_FIRST_PROTOCOL.md', ROOT/'docs'/'RESURFACE_MORE_PROTOCOL.md']
    return {str(p.relative_to(ROOT)): data.sha_file(p) for p in paths}


def equal_tree(actual, expected, where='state'):
    """Verify values and dtypes recursively; CPU/GPU residency may differ."""
    if isinstance(expected, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not torch.equal(actual.detach().cpu(), expected.detach().cpu())):
            raise RuntimeError(f'Exact tensor restoration failed: {where}')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f'Exact dictionary restoration failed: {where}')
        for key in expected:
            equal_tree(actual[key], expected[key], f'{where}.{key}')
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise RuntimeError(f'Exact sequence restoration failed: {where}')
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal_tree(a, b, f'{where}[{i}]')
    elif type(actual) is not type(expected) or actual != expected:
        raise RuntimeError(f'Exact scalar restoration failed: {where}')
    return True


def optimizer_names(bank):
    return [[n for n in bank.masters if n.endswith(('.V_read', '.g_read'))],
            [n for n in bank.masters if n.endswith(('.router_w', '.router_b'))]]


def check_optimizer(optimizer_state, masters, steps, factor):
    groups, states = optimizer_state['param_groups'], optimizer_state['state']
    names = [[n for n in masters if n.endswith(('.V_read', '.g_read'))],
             [n for n in masters if n.endswith(('.router_w', '.router_b'))]]
    if len(groups) != 2 or len(states) != 224 or any(len(ns) != 112 for ns in names):
        raise ValueError('Wrong resumed Adam parameter inventory')
    all_ids = []
    for group, ns, base_lr in zip(groups, names, (1e-4, 3e-4)):
        if (len(group['params']) != len(ns) or group['base_lr'] != base_lr
                or not math.isclose(group['lr'], base_lr * factor, rel_tol=1e-14)
                or group['betas'] != (.9, .999) or group['eps'] != 1e-8
                or group['weight_decay'] != 0. or group['amsgrad'] is not False
                or group['maximize'] is not False or group['differentiable'] is not False):
            raise ValueError('Resumed Adam parameter groups differ from frozen recipe')
        for index, name in zip(group['params'], ns):
            all_ids.append(index)
            value = states[index]
            if set(value) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Unexpected Adam parameter state')
            if value['step'].numel() != 1 or value['step'].item() != steps:
                raise ValueError('Adam successful step count differs')
            for field in ('exp_avg', 'exp_avg_sq'):
                tensor = value[field]
                if (tensor.dtype != torch.float32 or tensor.shape != masters[name].shape
                        or not bool(torch.isfinite(tensor).all())):
                    raise ValueError('Invalid Adam moments: ' + name)
            if not bool((value['exp_avg_sq'] >= 0).all()):
                raise ValueError('Adam variance is negative')
    if len(set(all_ids)) != 224 or set(all_ids) != set(states):
        raise ValueError('Adam parameters have missing/duplicate states')
    return names


def load_parent(args, tokenizer, old_binding):
    if data.sha_file(ROOT/'docs'/'RESURFACE_MORE_PROTOCOL.md') != PROTOCOL_SHA:
        raise ValueError('Continuation protocol changed')
    if data.sha_file(ROOT/'docs'/'QUANT_FIRST_PROTOCOL.md') != PARENT_PROTOCOL_SHA:
        raise ValueError('Parent protocol changed')
    if (data.sha_file(args.parent_training_report) != PARENT_REPORT_SHA
            or data.sha_file(args.parent_checkpoint) != PARENT_CHECKPOINT_SHA
            or data.sha_file(args.calibration) != CALIBRATION_SHA):
        raise ValueError('Parent report/checkpoint/calibration identity changed')
    parent_args = argparse.Namespace(calibration=args.calibration,
                                     training_report=args.parent_training_report)
    _, _, parent, adapter_path = load_parent_candidate(parent_args, tokenizer)
    if parent['binding'] != old_binding or data.sha_file(adapter_path) != PARENT_ADAPTER_SHA:
        raise ValueError('Parent training binding/adapter differs')
    for relative, expected in parent['code_sha256'].items():
        if data.sha_file(ROOT/relative) != expected:
            raise ValueError('Parent execution code changed: ' + relative)
    if parent['checkpoints'][-1]['sha256'] != PARENT_CHECKPOINT_SHA:
        raise ValueError('Parent report does not identify this final checkpoint')
    checkpoint = torch.load(args.parent_checkpoint, map_location='cpu', weights_only=True)
    if (checkpoint['format'] != 'MAMBA2_SQ_FIRST_RESURFACE_CHECKPOINT_V1'
            or checkpoint['binding'] != parent['binding']
            or checkpoint['successful_updates'] != PARENT_STEPS
            or checkpoint['attempts'] != parent['attempts']
            or checkpoint['scaler'] != EXPECTED_SCALER):
        raise ValueError('Parent checkpoint provenance/update/scaler differs')
    adapter = native.read_fp16(adapter_path, expected_binding=parent['binding'])
    if set(checkpoint['masters']) != set(adapter['tensors']) or len(checkpoint['masters']) != 224:
        raise ValueError('Parent FP32 master inventory differs from FP16 export')
    for name, value in checkpoint['masters'].items():
        if (value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                or not torch.equal(value.half(), adapter['tensors'][name])):
            raise ValueError('Parent FP32 master cast differs from export: ' + name)
    check_optimizer(checkpoint['optimizer'], checkpoint['masters'], PARENT_STEPS, .1)
    return parent, checkpoint, adapter_path


def pair_for(r, ordered, examples, windows, prose_order):
    row = examples[ordered[r]]
    full = torch.tensor(row['full_ids'], dtype=torch.long, device='cuda')[None]
    answer_mask = torch.zeros_like(full[:, :-1], dtype=torch.bool)
    answer_mask[:, row['ce_hidden_positions']] = True
    if int(answer_mask.sum()) != row['answer_target_count']:
        raise ValueError('Numeric answer suffix mask differs')
    j = PARENT_STEPS + r
    prose_index, prose_start = prose_order[j % 448], 512 * ((j // 448) % 4)
    prose = windows[prose_index, prose_start:prose_start+512].to('cuda')[None]
    if tuple(prose.shape) != (1, 512):
        raise ValueError('Prose segment geometry differs')
    return {'id': row['id'], 'schedule_entry': ordered[r], 'global_prose_index': j,
            'prose_window': prose_index, 'prose_start': prose_start,
            'mk_ids': full[:, :-1], 'mk_targets': full[:, 1:], 'answer_mask': answer_mask,
            'prose_ids': prose[:, :-1], 'prose_targets': prose[:, 1:]}


def expected_scaler_after(before, overflow):
    after = dict(before)
    if overflow:
        # Preserve v2 manual update(new_scale=scale/2): growth tracker is unchanged.
        after['scale'] *= .5
    else:
        after['_growth_tracker'] += 1
        if after['_growth_tracker'] == after['growth_interval']:
            after['scale'] *= after['growth_factor']
            after['_growth_tracker'] = 0
    return after


def attempt(bank, teacher, teacher_execution, pair, optimizer, scaler, r):
    optimizer.zero_grad(set_to_none=True)
    factor = lr_factor(r)
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * factor
    before = dict(scaler.state_dict())
    hidden, gates = bank.forward_hidden(pair['mk_ids'], use_checkpoint=True)
    mk = objective.staged_loss_backward(hidden, pair['mk_targets'], bank.model.lm_head,
        gates=gates, scaler=scaler, answer_mask=pair['answer_mask'], ce_weight=1.,
        kl_weight=0., closure_weight=0., chunk_tokens=64)
    del hidden, gates
    with torch.no_grad():
        teacher_hidden = teacher_execution.backbone(pair['prose_ids'], reset=True)
    hidden, gates = bank.forward_hidden(pair['prose_ids'], use_checkpoint=True)
    prose = objective.staged_loss_backward(hidden, pair['prose_targets'], bank.model.lm_head,
        gates=gates, scaler=scaler, teacher_hidden=teacher_hidden,
        teacher_head=teacher.lm_head, ce_weight=.5, kl_weight=.5,
        closure_weight=3., chunk_tokens=64)
    del hidden, gates, teacher_hidden
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in bank.parameters()]
    if any(g is None for g in gradients):
        raise RuntimeError('Missing adapter gradient')
    finite = all(bool(torch.isfinite(g).all()) for g in gradients)
    overflow = (not finite or not mk['scaled_hidden_gradient_finite']
                or not prose['scaled_hidden_gradient_finite'])
    if overflow:
        scaler.update(new_scale=scaler.get_scale()*.5)
        magnitude = None
    else:
        magnitude = float(torch.nn.utils.clip_grad_norm_(bank.parameters(), 1., error_if_nonfinite=True))
        scaler.step(optimizer)
        scaler.update()
        if any(not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(p.half()).all())
               for p in bank.parameters()):
            raise FloatingPointError('Nonfinite updated adapter master or FP16 cast')
    after = dict(scaler.state_dict())
    equal_tree(after, expected_scaler_after(before, overflow), 'scaler transition')
    bank.assert_base_frozen()
    return {'mk_ce': mk['ce'], 'prose_ce': prose['ce'],
            'prose_kl': prose['teacher_to_student_kl'], 'prose_closure': prose['closure'],
            'overflow': overflow, 'all_adapter_gradients_finite': finite,
            'mk_scaled_hidden_gradient_finite': mk['scaled_hidden_gradient_finite'],
            'prose_scaled_hidden_gradient_finite': prose['scaled_hidden_gradient_finite'],
            'gradient_norm_before_clip': magnitude, 'loss_scale_before': before['scale'],
            'loss_scale': after['scale'], 'scaler_before': before, 'scaler_after': after,
            'lr_factor': factor, 'learning_rates': [g['lr'] for g in optimizer.param_groups]}


def save_checkpoint(path, bank, optimizer, scaler, binding, success, attempts):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.pt.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    masters, optimizer_state = bank.state_dict(), optimizer.state_dict()
    mapping = check_optimizer(optimizer_state, masters, PARENT_STEPS+success, lr_factor(success-1))
    with temporary.open('wb') as stream:
        torch.save({'format': CHECKPOINT_FORMAT, 'binding': binding,
                    'additional_successful_updates': success,
                    'cumulative_successful_updates': PARENT_STEPS+success,
                    'successful_updates': PARENT_STEPS+success,
                    'additional_attempts': attempts, 'attempts': attempts,
                    'masters': masters, 'optimizer': optimizer_state,
                    'optimizer_parameter_names': mapping, 'scaler': scaler.state_dict()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return {'path': str(path), 'file': path.name, 'bytes': path.stat().st_size,
            'sha256': data.sha_file(path), 'additional_successful_updates': success,
            'cumulative_successful_updates': PARENT_STEPS+success}


def validate_smoke(path, binding, hashes, output_dir):
    if path is None:
        raise ValueError('Formal continuation requires --smoke-report from a discarded update')
    smoke = json.loads(path.read_text())
    if (path.parent.resolve() == output_dir.resolve() or smoke.get('format') != FORMAT
            or smoke.get('mode') != 'smoke' or smoke.get('complete') is not True
            or smoke.get('additional_successful_updates') != 1
            or smoke.get('cumulative_successful_updates') != PARENT_STEPS+1
            or smoke.get('binding') != binding or smoke.get('code_sha256') != hashes
            or smoke.get('checkpoints') != [] or 'adapter' in smoke
            or smoke.get('parent_resume_check', {}).get('packed_training_forward_bitwise_equal') is not True
            or smoke.get('frozen_base_check', {}).get('identity_version_gradients_unchanged') is not True
            or smoke.get('frozen_state_calibration_check') is not True
            or not 1 <= smoke.get('additional_attempts', 0) <= 9):
        raise ValueError('Discarded resumed smoke is incomplete or differs from formal inputs/code')
    return {'path': str(path), 'sha256': data.sha_file(path), 'discarded': True,
            'additional_successful_updates': 1, 'parent_reloaded_for_formal': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--train-manifest-sha256', required=True)
    p.add_argument('--prose-manifest', type=Path, required=True)
    p.add_argument('--prose-tokens', type=Path, required=True)
    p.add_argument('--parent-training-report', type=Path, required=True)
    p.add_argument('--parent-checkpoint', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--smoke-report', type=Path)
    p.add_argument('--smoke', action='store_true', help='Discard one resumed update; no export/checkpoint')
    args = p.parse_args()
    if args.smoke and args.smoke_report is not None:
        p.error('--smoke-report is only used for the formal continuation')
    if args.out_dir.exists():
        raise FileExistsError('New output directory required')
    args.out_dir.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.manual_seed(2026092804); torch.cuda.manual_seed_all(2026092804)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    hashes = code_hashes()
    report = {'format': FORMAT, 'complete': False, 'mode': 'smoke' if args.smoke else 'formal',
              'started_unix': time.time(), 'source_sha256': data.sha_file(__file__),
              'code_sha256': hashes, 'continuation_protocol_sha256': PROTOCOL_SHA,
              'parent_successful_updates': PARENT_STEPS, 'planned_additional_successful_updates': STEPS,
              'planned_cumulative_successful_updates': PARENT_STEPS+STEPS,
              'history': [], 'checkpoints': []}
    bank = student_execution = teacher_execution = None
    try:
        tokenizer, examples, windows, prose_order, old_binding, permutations = load_training_inputs(args)
        parent, checkpoint, parent_adapter = load_parent(args, tokenizer, old_binding)
        binding = dict(old_binding, continuation_protocol_sha256=PROTOCOL_SHA,
                       parent_training_report_sha256=PARENT_REPORT_SHA,
                       parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA,
                       parent_adapter_sha256=PARENT_ADAPTER_SHA,
                       parent_successful_updates=PARENT_STEPS, additional_successful_updates=STEPS,
                       successful_updates=PARENT_STEPS+STEPS, initial_adapter=INITIAL_ADAPTER)
        report['binding'] = binding
        report['parent'] = {'training_report': str(args.parent_training_report),
            'training_report_sha256': PARENT_REPORT_SHA, 'checkpoint': str(args.parent_checkpoint),
            'checkpoint_sha256': PARENT_CHECKPOINT_SHA, 'adapter': str(parent_adapter),
            'adapter_sha256': PARENT_ADAPTER_SHA, 'attempts': parent['attempts']}
        if not args.smoke:
            report['discarded_smoke'] = validate_smoke(args.smoke_report, binding, hashes, args.out_dir)
        write_json(args.out_dir/'report.json', report)
        print('[setup] loading frozen source student and independent S16 teacher', flush=True)
        student = runtime.load_source_model(args.source_dir)
        teacher = runtime.load_source_model(args.source_dir)
        probe_ids = windows[0, :128].cuda()[None]
        with torch.no_grad(), native.install_fp16(student, parent_adapter, expected_binding=parent['binding']):
            with StateQuant(student, 'sq3p25', permutations) as packed:
                parent_hidden = packed.backbone(probe_ids, reset=True)
                parent_cache = packed.cache_breakdown()
        if parent_cache['total_bytes'] != CACHE_BYTES:
            raise RuntimeError('Parent packed cache changed')
        bank = native.ResurfaceNative(student, 'soft')
        bank.load_state_dict(checkpoint['masters'])
        optimizer = optimizer_for(bank)
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        scaler = torch.amp.GradScaler('cuda', init_scale=1024., growth_factor=2.,
                                      backoff_factor=.5, growth_interval=2000)
        scaler.load_state_dict(copy.deepcopy(checkpoint['scaler']))
        equal_tree(bank.state_dict(), checkpoint['masters'], 'masters')
        equal_tree(optimizer.state_dict(), checkpoint['optimizer'], 'optimizer')
        equal_tree(scaler.state_dict(), checkpoint['scaler'], 'scaler')
        report['optimizer_parameter_names'] = optimizer_names(bank)
        student_execution = StateQuantTraining(student, permutations).install()
        teacher_execution = StateQuant(teacher, 's16').install()
        with torch.no_grad():
            restored_hidden, _ = bank.forward_hidden(probe_ids, use_checkpoint=False)
        if not torch.equal(parent_hidden, restored_hidden):
            raise RuntimeError('Restored training forward differs from parent packed FP16 inference')
        report['parent_resume_check'] = {'masters_exact': True, 'optimizer_exact': True,
            'scaler_exact': True, 'master_fp16_cast_equals_parent_export': True,
            'optimizer_steps': PARENT_STEPS, 'optimizer_states': 224, 'master_tensors': 224,
            'scaler': scaler.state_dict(), 'packed_training_forward_bitwise_equal': True,
            'probe_tokens': 128, 'probe_token_sha256': runtime.token_digest(probe_ids.cpu().numpy()),
            'cache': parent_cache}
        del parent_hidden, restored_hidden, probe_ids, checkpoint
        print('[setup] exact parent masters/Adam/scaler restored; 128-token packed forward matches', flush=True)
        ordered = schedule()
        report['schedule'] = {'numeric_seeds': [2026092804, 2026092805],
            'numeric_order_sha256_int64le': runtime.token_digest(torch.tensor(ordered, dtype=torch.int64).numpy()),
            'prose_global_start': PARENT_STEPS, 'additional_updates': STEPS}
        successful = attempts = overflows = 0
        limit = 1 if args.smoke else STEPS
        while successful < limit:
            pair = pair_for(successful, ordered, examples, windows, prose_order)
            r = successful
            start = time.monotonic()
            result = attempt(bank, teacher, teacher_execution, pair, optimizer, scaler, r)
            student_execution.assert_frozen()
            attempts += 1
            overflows += int(result['overflow'])
            successful += int(not result['overflow'])
            row = {'attempt': attempts, 'additional_update_index': r,
                   'additional_successful_updates': successful,
                   'cumulative_successful_updates': PARENT_STEPS+successful,
                   'successful_updates': PARENT_STEPS+successful,
                   'case_id': pair['id'], 'schedule_entry': pair['schedule_entry'],
                   'global_prose_index': pair['global_prose_index'],
                   'prose_window': pair['prose_window'], 'prose_start': pair['prose_start'],
                   'answer_targets': int(pair['answer_mask'].sum()),
                   'seconds': time.monotonic()-start, **result}
            report['history'].append(row)
            report.update(additional_successful_updates=successful,
                cumulative_successful_updates=PARENT_STEPS+successful, successful_updates=PARENT_STEPS+successful,
                additional_attempts=attempts, attempts=attempts, additional_overflows=overflows)
            if overflows > 8:
                raise RuntimeError('Additional overflow retry budget exceeded')
            if not args.smoke and successful and successful % 768 == 0 and not result['overflow']:
                report['checkpoints'].append(save_checkpoint(args.out_dir/f'checkpoint_{successful:04d}.pt',
                    bank, optimizer, scaler, binding, successful, attempts))
            if attempts % 16 == 0 or successful == limit or result['overflow']:
                report['gpu_memory'] = runtime.gpu_memory_receipt()
                write_json(args.out_dir/'report.json', report)
                print(json.dumps({'additional_updates': successful, 'cumulative_updates': PARENT_STEPS+successful,
                    'attempts': attempts, 'mk_ce': result['mk_ce'], 'prose_ce': result['prose_ce'],
                    'overflow': result['overflow'], 'loss_scale': result['loss_scale']}), flush=True)
        report['frozen_base_check'] = bank.assert_base_frozen()
        report['frozen_state_calibration_check'] = student_execution.assert_frozen()
        report['teacher_base_parameters_frozen'] = all(not p.requires_grad and p.grad is None
                                                       for p in teacher.parameters())
        report['final_scaler'] = scaler.state_dict()
        check_optimizer(optimizer.state_dict(), bank.state_dict(), PARENT_STEPS+successful, lr_factor(successful-1))
        if not args.smoke:
            adapter_path = args.out_dir/'adapter_fp16.pt'
            report['adapter'] = bank.export_fp16(adapter_path, binding=binding)
            actual_adapter = native.read_fp16(adapter_path, expected_binding=binding)
            final_checkpoint = torch.load(args.out_dir/'checkpoint_3072.pt', map_location='cpu', weights_only=True)
            equal_tree(final_checkpoint['masters'], bank.state_dict(), 'final checkpoint masters')
            equal_tree(final_checkpoint['optimizer'], optimizer.state_dict(), 'final checkpoint optimizer')
            equal_tree(final_checkpoint['scaler'], scaler.state_dict(), 'final checkpoint scaler')
            for name, value in final_checkpoint['masters'].items():
                if not torch.equal(value.half(), actual_adapter['tensors'][name]):
                    raise RuntimeError('Final checkpoint master cast differs from exported FP16 tensor')
            report['final_checkpoint_export_check'] = {'all_master_casts_equal_export': True,
                'master_tensors': 224, 'optimizer_exact': True, 'scaler_exact': True,
                'final_checkpoint_sha256': report['checkpoints'][-1]['sha256']}
            export_probe = windows[0, :128].cuda()[None]
            with torch.no_grad():
                training_hidden, _ = bank.forward_hidden(export_probe, use_checkpoint=False)
            student_execution.close(); bank.close()
            with torch.no_grad(), native.install_fp16(student, adapter_path, expected_binding=binding):
                with StateQuant(student, 'sq3p25', permutations) as deployed:
                    deployed_hidden = deployed.backbone(export_probe, reset=True)
                    deploy_cache = deployed.cache_breakdown()
            if not torch.equal(training_hidden, deployed_hidden) or deploy_cache != parent_cache:
                raise RuntimeError('Final packed training/deployment forward or cache differs')
            report['deployed_export_check'] = {'packed_training_forward_bitwise_equal': True,
                'probe_tokens': 128, 'probe_token_sha256': runtime.token_digest(export_probe.cpu().numpy()),
                'cache': deploy_cache, 'cache_unchanged_from_parent': True}
        report.update(complete=True, gpu_memory=runtime.gpu_memory_receipt())
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for resource in (student_execution, teacher_execution, bank):
            if resource is not None:
                resource.close()
        report['finished_unix'] = time.time()
        write_json(args.out_dir/'report.json', report)


if __name__ == '__main__':
    main()
