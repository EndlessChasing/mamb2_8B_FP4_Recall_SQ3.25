#!/usr/bin/env python3
"""Train fresh Resurface on frozen W4 weights and TRAIN-selected Q3.25 state.

The deployed V10 scan, V11 STE and V2 optimization mathematics remain unchanged.
This independent experiment binds actual W4 files, fresh calibration, audited
pretraining controls, discarded smoke and the final-only FP16 export.
"""
from __future__ import annotations

import argparse
import copy
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
from mamba2_recall import resurface_data as data, resurface_native as native, runtime
from mamba2_recall.state_quant import StateQuant
from state_ppl_training_v11 import StateQuantTrainingV11
from state_ppl_codec_v10 import StatePPLQuantV10
from train_quant_first import (STEPS, PROSE_MANIFEST_SHA, PROSE_TOKENS_SHA,
    attempt, lr_factor, optimizer_for, pair_for, schedule, write_json)
from train_resurface_more import (check_optimizer, equal_tree,
    expected_scaler_after, optimizer_names)
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
import w4_state_binding_v1 as experiment
from w4_state_binding_v1 import (PROTOCOL_SHA, W4_MANIFEST_SHA, LAYOUT,
    CACHE_BYTES, NUMERIC_PROTOCOL_SHA, TRAIN_MANIFEST_SHA, need, read_json)

V2_TRAINER_SHA = '547e5b64904a273cd96a7d77624e5225a67edb9fa683950a43efc2d4171392a6'
FORMAT = 'W4_STATE_RESURFACE_TRAIN_V1'
CHECKPOINT_FORMAT = 'W4_STATE_RESURFACE_CHECKPOINT_V1'
PROOF_FORMAT = 'W4_STATE_RESURFACE_TRAIN_PROOF_V1'
INITIAL_ADAPTER = 'fresh V=0,g=1,w=0,b=-4; no pretrained adapter or checkpoint'
TEACHER = 'separate unadapted W4 package, S16 per-token carry'
INITIAL_SCALER = {'scale': 1024., 'growth_factor': 2., 'backoff_factor': .5,
                  'growth_interval': 2000, '_growth_tracker': 0}
BASE_PARAMETER_BYTES = 16473999360
TRAINING_FILES = (
    'scripts/train_w4_state_resurface_v1.py',
    'scripts/train_quant_first.py', 'scripts/train_resurface_more.py',
    'scripts/evaluate_resurface_more.py',
    'scripts/state_ppl_training_v11.py', 'scripts/state_ppl_training_scan_v11.py',
    'mamba2_recall/state_training.py', 'mamba2_recall/state_training_scan.py',
    'mamba2_recall/resurface_native.py', 'mamba2_recall/resurface_loss.py',
    'mamba2_recall/resurface_data.py', 'docs/QUANT_FIRST_PROTOCOL.md',
    'docs/RESURFACE_MORE_BACKEND_REPLAY.md')


def code_hashes():
    """Training plus shared dependencies, independent of evaluator/auditor edits."""
    return experiment.code_hashes(extra=TRAINING_FILES)


def require_cpu_audit(path, report_path, stage):
    """Bind an independently completed audit without executing auditor code."""
    need(path is not None, 'A completed independent CPU audit is required')
    audit = read_json(path)
    need(audit.get('format') == 'W4_STATE_AUDIT_V1'
         and audit.get('complete') is True and audit.get('passed') is True
         and audit.get('cuda_initialized') is False and audit.get('stage') == stage
         and audit.get('protocol_sha256') == PROTOCOL_SHA
         and audit.get('input_report_sha256') == data.sha_file(report_path)
         and audit.get('source_sha256') == data.sha_file(ROOT/'scripts/audit_w4_state_v1.py'),
         'Independent CPU audit is incomplete or bound to different input/source')
    return dict(sha256=data.sha_file(path), input_report_sha256=data.sha_file(report_path),
                stage=stage, complete=True, passed=True, cuda_initialized=False)


def validate_pretrain(report_path, audit_path, parent_binding):
    """Require frozen, complete matched controls; PPL never changes selection."""
    report = read_json(report_path)
    need(report.get('format') == 'W4_STATE_PRETRAIN_V1'
         and report.get('complete') is True
         and report.get('protocol_sha256') == PROTOCOL_SHA
         and report.get('w4_manifest_sha256') == W4_MANIFEST_SHA
         and report.get('calibration_sha256') == parent_binding['calibration_sha256'],
         'Complete W4/S16 and selected W4/SQ pretraining controls required')
    inventory = report.get('code_sha256')
    need(isinstance(inventory, dict) and inventory, 'Pretraining code inventory is absent')
    for relative, digest in inventory.items():
        source = (ROOT/relative).resolve()
        need(isinstance(relative, str) and not Path(relative).is_absolute()
             and source.is_relative_to(ROOT.resolve()) and source.is_file()
             and data.sha_file(source) == digest, 'Pretraining source changed: '+str(relative))
    for relative, digest in experiment.code_hashes().items():
        need(inventory.get(relative) == digest, 'Pretraining shared dependency differs: '+relative)
    reports = report.get('reports', {})
    need(set(reports) == {'w4_s16', 'w4_sq325'}, 'Pretraining control inventory differs')
    raw_hashes = {}
    for arm, item in reports.items():
        filename = item.get('file')
        need(isinstance(filename, str) and Path(filename).name == filename,
             'Pretraining arm must use a local filename')
        raw_path = report_path.parent/filename
        need(data.sha_file(raw_path) == item.get('sha256'), 'Pretraining raw arm changed: '+arm)
        raw = read_json(raw_path)
        ppl = raw.get('ppl', {}); rows = ppl.get('windows', [])
        need(raw.get('complete') is True and len(rows) == 130
             and sum(row['target_tokens'] for row in rows) == 264764
             and ppl.get('target_tokens') == 264764
             and all(math.isfinite(row['nll']) and row['nll'] >= 0 for row in rows)
             and ppl.get('nll') == sum(row['nll'] for row in rows)
             and math.isfinite(ppl.get('ppl', float('nan')))
             and ppl['ppl'] == math.exp(ppl['nll']/264764),
             'Pretraining full PPL arithmetic or completeness differs: '+arm)
        raw_hashes[arm] = item['sha256']
    proof = require_cpu_audit(audit_path, report_path, 'pretrain')
    return dict(pretrain_report_sha256=data.sha_file(report_path),
                pretrain_audit_sha256=proof['sha256'], pretrain_raw_report_sha256=raw_hashes)


def load_inputs(args):
    experiment.check_protocol()
    need(data.sha_file(ROOT/'docs/QUANT_FIRST_PROTOCOL.md') == NUMERIC_PROTOCOL_SHA,
         'Original numeric TRAIN protocol changed')
    need(data.sha_file(ROOT/'scripts/train_quant_first.py') == V2_TRAINER_SHA,
         'Reused V2 training mathematics changed')
    need(args.train_manifest_sha256 == TRAIN_MANIFEST_SHA, 'Wrong numeric TRAIN manifest')
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    manifest, _, examples = data.load_training(args.data_root, TRAIN_MANIFEST_SHA, tokenizer)
    need(manifest['protocol_sha256'] == NUMERIC_PROTOCOL_SHA and len(examples) == STEPS,
         'Numeric TRAIN provenance or inventory differs')
    need(data.sha_file(args.prose_manifest) == PROSE_MANIFEST_SHA
         and data.sha_file(args.prose_tokens) == PROSE_TOKENS_SHA,
         'Pinned prose TRAIN files differ')
    prose_manifest = read_json(args.prose_manifest)
    windows = torch.load(args.prose_tokens, map_location='cpu', weights_only=True)
    need(prose_manifest.get('complete') is True
         and prose_manifest.get('training_tokens_file_sha256') == PROSE_TOKENS_SHA
         and tuple(windows.shape) == (448, 2048) and windows.dtype == torch.int64
         and sorted(prose_manifest['schedule']) == list(range(448))
         and runtime.token_digest(windows.flatten().numpy()) ==
             prose_manifest['training_tokens_sha256_int64le'],
         'Expected 448 pinned prose TRAIN windows')
    calibration, receipt, parent_binding = experiment.load_selection(args.calibration)
    pretrain_binding = validate_pretrain(args.pretrain_report, args.pretrain_audit, parent_binding)
    kernel_proof = experiment.validate_upstream_kernels()
    binding = {**parent_binding, **pretrain_binding,
        'upstream_kernel_proof': kernel_proof,
        'initial_adapter': INITIAL_ADAPTER, 'fresh_initialization': True,
        'prior_adapter_loaded': False, 'checkpoint_loaded': False,
        'v2_trainer_sha256': V2_TRAINER_SHA,
        'state_mode': 'v10_32_32_64; exact packed stored-scale forward; 64-live STE backward',
        'teacher': TEACHER, 'teacher_w4_manifest_sha256': W4_MANIFEST_SHA,
        'source_checkpoint_sha256': runtime.SOURCE_CHECKPOINT_SHA256,
        'tokenizer_sha256': tokenizer.sha256, 'protocol_sha256': PROTOCOL_SHA,
        'numeric_protocol_sha256': NUMERIC_PROTOCOL_SHA,
        'train_manifest_sha256': TRAIN_MANIFEST_SHA,
        'prose_manifest_sha256': PROSE_MANIFEST_SHA, 'prose_tokens_sha256': PROSE_TOKENS_SHA,
        'prose_tokens_int64le_sha256': prose_manifest['training_tokens_sha256_int64le'],
        'adapter': native.FORMAT, 'successful_updates': STEPS}
    return tokenizer, examples, windows, prose_manifest['schedule'], binding, calibration['permutations']


def frozen_identity(model, identities):
    values = dict(model.named_parameters())
    need(set(values) == set(identities) and len(values) == 507,
         'Frozen W4 parameter inventory changed')
    for name, value in values.items():
        need((id(value), value.data_ptr(), value._version) == identities[name]
             and value.dtype == torch.float16 and not value.requires_grad and value.grad is None,
             'Frozen W4 identity/version/dtype/gradient changed: '+name)
    return True


def full_loaded_check(model, hashes, identities):
    """Hash actual loaded FP16 bytes against the independent W4 decoded ledger."""
    values = dict(model.named_parameters())
    need(set(values) == set(hashes) and len(values) == 507,
         'W4 decoded hash ledger does not cover every parameter')
    frozen_identity(model, identities)
    for name, value in values.items():
        need(value.dtype == torch.float16 and not value.requires_grad and value.grad is None
             and native.tensor_hash(value) == hashes[name], 'Loaded W4 bytes differ: '+name)
    size = sum(v.numel()*v.element_size() for v in values.values())
    need(size == BASE_PARAMETER_BYTES, 'Loaded W4 parameter byte count differs')
    return dict(tensors=507, parameters=8236999680,
        identity_version_gradients_unchanged=True, actual_content_checked=True,
        w4_manifest_sha256=W4_MANIFEST_SHA, loaded_parameter_bytes=size,
        decoded_tensor_sha256=dict(sorted(hashes.items())))


def bank_full_check(bank, hashes):
    proof = bank.assert_base_frozen(check_values=True)
    size = sum(p.numel()*p.element_size() for p in bank.model.parameters())
    need(size == BASE_PARAMETER_BYTES, 'Loaded W4 parameter byte count differs')
    return dict(proof, w4_manifest_sha256=W4_MANIFEST_SHA, loaded_parameter_bytes=size,
                decoded_tensor_sha256=dict(sorted(hashes.items())))


def restored_native(model, originals):
    mixers = [layer.mixer for layer in model.backbone.layers]
    need(len(mixers) == len(originals) == 56
         and all(mx.forward == original and not mx._forward_pre_hooks
                 and not mx._forward_hooks and not mx.norm._forward_pre_hooks
                 for mx, original in zip(mixers, originals)),
         'State controller or Resurface failed to restore native mixer methods/hooks')
    return dict(native_forwards_restored=True, adapter_hooks_removed=True, layers=56)


def assert_fresh(bank, optimizer, scaler):
    if len(bank.masters) != 224 or sum(p.numel() for p in bank.parameters()) != 1154104:
        raise ValueError('Wrong fresh adapter geometry')
    for name, parameter in bank.masters.items():
        field = name.rsplit('.', 1)[-1]
        expected = {'V_read': 0., 'g_read': 1., 'router_w': 0., 'router_b': -4.}[field]
        if (parameter.dtype != torch.float32 or not parameter.requires_grad
                or parameter.grad is not None or not bool((parameter == expected).all())):
            raise ValueError('Adapter was not initialized freshly: '+name)
    if optimizer.state or optimizer.state_dict()['state']:
        raise ValueError('Fresh optimizer already has moments or steps')
    if any(len(group['params']) != 112 for group in optimizer.param_groups):
        raise ValueError('Wrong fresh optimizer group inventory')
    equal_tree(scaler.state_dict(), INITIAL_SCALER, 'fresh GradScaler')
    return {'all_224_masters_exact_fresh_values': True, 'masters_dtype': 'float32',
        'master_tensors': 224, 'parameters': 1154104,
        'initial_tensor_sha256': {n: native.tensor_hash(p) for n, p in bank.masters.items()},
        'optimizer_state_empty': True, 'optimizer_steps': 0,
        'scaler_exact_initial': True, 'scaler': scaler.state_dict(),
        'prior_adapter_loaded': False, 'checkpoint_loaded': False}


def finite_packed(execution, hidden):
    tensors = [hidden]
    for layer in execution._cache:
        tensors.append(layer.conv)
        tensors.extend(v for v in layer.state.tensors.values() if v.is_floating_point())
    if not bool(torch.stack([torch.isfinite(v).all() for v in tensors]).all()):
        raise FloatingPointError('Nonfinite packed probe output, convolution state or scale')
    return True


def probe_record(tokens, packed_hidden, training_hidden, cache):
    if not torch.equal(packed_hidden.cpu(), training_hidden.cpu()):
        raise RuntimeError('Training and deployed probe differ')
    if not bool(torch.isfinite(packed_hidden).all()):
        raise FloatingPointError('Nonfinite parity probe')
    return {'probe_tokens': tokens.shape[1],
        'probe_token_sha256': runtime.token_digest(tokens.cpu().numpy()),
        'packed_hidden_sha256': native.tensor_hash(packed_hidden),
        'training_hidden_sha256': native.tensor_hash(training_hidden),
        'packed_training_forward_bitwise_equal': True, 'packed_probe_finite': True,
        'cache': cache}


def save_probe_evidence(path, evidence):
    if path.exists():
        raise FileExistsError(path)
    with path.open('wb') as stream:
        torch.save(evidence, stream)
        stream.flush(); os.fsync(stream.fileno())
    return dict(file=path.name, sha256=data.sha_file(path), bytes=path.stat().st_size)


def save_checkpoint(path, bank, optimizer, scaler, binding, successful, attempts):
    if path.exists() or path.with_suffix('.pt.tmp').exists():
        raise FileExistsError(path)
    masters, optimizer_state = bank.state_dict(), optimizer.state_dict()
    mapping = check_optimizer(optimizer_state, masters, successful, lr_factor(successful-1))
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('wb') as stream:
        torch.save({'format': CHECKPOINT_FORMAT, 'binding': binding,
            'successful_updates': successful, 'attempts': attempts,
            'masters': masters, 'optimizer': optimizer_state,
            'optimizer_parameter_names': mapping, 'scaler': scaler.state_dict()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return {'path': str(path), 'file': path.name, 'sha256': data.sha_file(path),
        'bytes': path.stat().st_size, 'successful_updates': successful, 'attempts': attempts}


def validate_smoke(path, binding, hashes, out_dir):
    if path is None:
        raise ValueError('Formal run requires a discarded one-update --smoke-report')
    report = json.loads(path.read_text())
    if (path.parent.resolve() == out_dir.resolve() or report.get('format') != FORMAT
            or report.get('mode') != 'smoke' or report.get('complete') is not True
            or report.get('successful_updates') != 1 or report.get('binding') != binding
            or report.get('code_sha256') != hashes or report.get('checkpoints') != []
            or 'adapter' in report or not 1 <= report.get('attempts', 0) <= 9
            or report.get('fresh_initialization', {}).get('all_224_masters_exact_fresh_values') is not True
            or report.get('fresh_initialization', {}).get('optimizer_state_empty') is not True
            or report.get('fresh_initialization', {}).get('scaler_exact_initial') is not True
            or report.get('fresh_initialization', {}).get('prior_adapter_loaded') is not False
            or report.get('fresh_initialization', {}).get('checkpoint_loaded') is not False
            or report.get('initialization_check', {}).get('fresh_identity_matches_packed_bitwise') is not True
            or report.get('deployed_export_check', {}).get('packed_training_forward_bitwise_equal') is not True
            or report.get('deployed_export_check', {}).get('probe_tokens') != 128
            or report.get('export_master_check', {}).get('all_master_casts_equal_export') is not True
            or report.get('frozen_base_check', {}).get('identity_version_gradients_unchanged') is not True
            or report.get('frozen_base_check', {}).get('actual_content_checked') is not True
            or report.get('initial_student_base_check', {}).get('actual_content_checked') is not True
            or report.get('initial_teacher_base_check', {}).get('actual_content_checked') is not True
            or report.get('final_teacher_base_check', {}).get('actual_content_checked') is not True
            or report.get('frozen_state_calibration_check') is not True
            or report.get('selected_table_bytes_unchanged') is not True
            or report.get('teacher_base_parameters_frozen') is not True
            or report.get('history', [{}])[-1].get('all_adapter_gradients_finite_after_attempt') is not True
            or report.get('history', [{}])[-1].get('overflow') is not False
            or report.get('backend_final_check', {}).get('singleton_config_unchanged') is not True):
        raise ValueError('Discarded smoke is incomplete or differs from formal inputs/code')
    for field in ('initialization_check', 'deployed_export_check'):
        probes = report[field].get('probes', [])
        need([row['probe_tokens'] for row in probes] == [128,512]
             and all(row['packed_training_forward_bitwise_equal'] is True for row in probes),
             'Smoke requires exact 128/512-token probes')
    for name in ('initial_student_base_check', 'initial_teacher_base_check',
                 'frozen_base_check', 'final_teacher_base_check'):
        endpoint = report[name]
        need(endpoint.get('w4_manifest_sha256') == W4_MANIFEST_SHA
             and endpoint.get('loaded_parameter_bytes') == BASE_PARAMETER_BYTES
             and endpoint.get('tensors') == 507 and endpoint.get('parameters') == 8236999680
             and len(endpoint.get('decoded_tensor_sha256', {})) == 507,
             'Smoke full W4 loaded-byte proof differs: '+name)
    need(report.get('native_restoration') == {
        role: dict(native_forwards_restored=True, adapter_hooks_removed=True, layers=56)
        for role in ('student', 'teacher')}, 'Smoke native methods/hooks were not restored')
    evidence = report['probe_evidence']
    need(isinstance(evidence.get('file'), str) and Path(evidence['file']).name == evidence['file'],
         'Smoke evidence must use a local filename')
    evidence_path = path.parent/evidence['file']
    need(data.sha_file(evidence_path) == evidence['sha256']
         and evidence_path.stat().st_size == evidence['bytes'], 'Smoke raw evidence differs')
    export = report.get('smoke_adapter', {})
    need(isinstance(export.get('file'), str) and Path(export['file']).name == export['file'],
         'Smoke adapter must use a local filename')
    adapter_path = path.parent/export.get('file', '')
    if export.get('discarded') is not True or not adapter_path.is_file():
        raise ValueError('Discarded smoke export is absent')
    if data.sha_file(adapter_path) != export['sha256'] or adapter_path.stat().st_size != export['bytes']:
        raise ValueError('Discarded smoke export changed')
    payload = native.read_fp16(adapter_path, expected_binding=binding)
    nonzero = sum(int(torch.count_nonzero(value)) for name, value in payload['tensors'].items()
        if name.endswith('.V_read'))
    need(nonzero > 0 and report.get('export_nonidentity_check') == dict(
        fp16_V_read_nonzero_elements=nonzero, fp16_V_read_nonzero=True),
        'Discarded smoke export is an identity adapter or its nonzero receipt differs')
    return {'path': str(path), 'sha256': data.sha_file(path), 'discarded': True,
        'successful_updates': 1, 'formal_reinitializes_masters_optimizer_scaler': True,
        'smoke_export_sha256': export['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w4-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--pretrain-report', type=Path, required=True)
    parser.add_argument('--pretrain-audit', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--train-manifest-sha256', required=True)
    parser.add_argument('--prose-manifest', type=Path, required=True)
    parser.add_argument('--prose-tokens', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--smoke-report', type=Path)
    parser.add_argument('--smoke-audit', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Discard one fresh update and its audit export')
    args = parser.parse_args()
    if args.smoke and (args.smoke_report is not None or args.smoke_audit is not None):
        parser.error('--smoke-report/--smoke-audit are only used for formal training')
    if not args.smoke and (args.smoke_report is None or args.smoke_audit is None):
        parser.error('Formal training requires --smoke-report and --smoke-audit')
    if args.out_dir.exists():
        raise FileExistsError('New output directory required')
    args.out_dir.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.manual_seed(2026092803); torch.cuda.manual_seed_all(2026092803)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    report = {'format': FORMAT, 'complete': False, 'mode': 'smoke' if args.smoke else 'formal',
        'started_unix': time.time(), 'source_sha256': data.sha_file(__file__),
        'code_sha256': code_hashes(), 'history': [], 'checkpoints': [],
        'planned_successful_updates': STEPS, 'successful_updates': 0, 'attempts': 0, 'overflows': 0}
    bank = student_execution = teacher_execution = None
    try:
        tokenizer, examples, windows, prose_order, binding, permutations = load_inputs(args)
        policy = pin_replay_backend()
        binding.update(training_code_sha256=report['code_sha256'], backend_policy=policy)
        report['binding'] = binding
        if not args.smoke:
            report['discarded_smoke_check'] = validate_smoke(args.smoke_report, binding,
                report['code_sha256'], args.out_dir)
            report['discarded_smoke_check']['independent_audit'] = require_cpu_audit(
                args.smoke_audit, args.smoke_report, 'training')
        if not args.smoke:
            smoke_policy = json.loads(args.smoke_report.read_text())['backend_policy']
            equal_tree(policy, smoke_policy, 'smoke/formal pinned backend policy')
            report['discarded_smoke_check']['backend_policy_exact'] = True
        report['backend_policy'] = policy
        report['environment'] = runtime.environment_receipt()
        write_json(args.out_dir/'report.json', report)
        print('[setup] loading frozen W4 student and separate unadapted W4+S16 teacher', flush=True)
        base_hashes = experiment.expected_base_hashes(args.w4_dir)
        torch.cuda.reset_peak_memory_stats()
        student = experiment.load_w4(args.w4_dir)
        teacher = experiment.load_w4(args.w4_dir)
        if sum(p.numel() for p in student.parameters()) != 8236999680:
            raise RuntimeError('Wrong source parameter count')
        teacher_parameters = dict(teacher.named_parameters())
        teacher_identity = {n: (id(p), p.data_ptr(), p._version) for n, p in teacher_parameters.items()}
        student_forwards = [layer.mixer.forward for layer in student.backbone.layers]
        teacher_forwards = [layer.mixer.forward for layer in teacher.backbone.layers]
        restored_native(student, student_forwards); restored_native(teacher, teacher_forwards)
        report['student_w4_package'] = student._package_receipt
        report['teacher_w4_package'] = teacher._package_receipt
        need(student._package_receipt['manifest_sha256'] == W4_MANIFEST_SHA
             and teacher._package_receipt['manifest_sha256'] == W4_MANIFEST_SHA,
             'Loaded student/teacher W4 manifest differs')
        report['initial_teacher_base_check'] = full_loaded_check(teacher, base_hashes, teacher_identity)
        if {p.data_ptr() for p in student.parameters()} & {p.data_ptr() for p in teacher.parameters()}:
            raise RuntimeError('Teacher and student source storage is not separate')
        proof = {'format': PROOF_FORMAT,
            'binding': binding, 'initialization': {}, 'export': {}}
        initial_caches = {}
        with torch.no_grad(), StatePPLQuantV10(student, permutations, layout=LAYOUT) as packed:
            for length in (128,512):
                probe = windows[0, :length].cuda()[None]
                reference_hidden = packed.backbone(probe, reset=True)
                initial_caches[length] = packed.cache_breakdown()
                finite_packed(packed, reference_hidden)
                if initial_caches[length]['total_bytes'] != CACHE_BYTES:
                    raise RuntimeError('Packed initial cache differs from frozen budget')
                proof['initialization'][str(length)] = dict(token_ids=probe.cpu(),
                    packed_hidden=reference_hidden.cpu())
        initial_cache = initial_caches[128]
        del reference_hidden, probe
        bank = native.ResurfaceNative(student, 'soft', expected_base_hashes=base_hashes)
        report['initial_student_base_check'] = bank_full_check(bank, base_hashes)
        optimizer = optimizer_for(bank)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024., growth_factor=2.,
            backoff_factor=.5, growth_interval=2000)
        report['fresh_initialization'] = assert_fresh(bank, optimizer, scaler)
        report['optimizer_parameter_names'] = optimizer_names(bank)
        student_execution = StateQuantTrainingV11(student, permutations).install()
        teacher_execution = StateQuant(teacher, 's16').install()
        initial_probes = []
        with torch.no_grad():
            for length in (128,512):
                entry = proof['initialization'][str(length)]
                probe = entry['token_ids'].cuda()
                identity_hidden, _ = bank.forward_hidden(probe, use_checkpoint=False)
                entry['training_hidden'] = identity_hidden.cpu()
                initial_probes.append(probe_record(probe, entry['packed_hidden'],
                    entry['training_hidden'], initial_caches[length]))
        report['initialization_check'] = {'fresh_identity_matches_packed_bitwise': True,
            'probe_tokens': 128, 'probe_token_sha256': initial_probes[0]['probe_token_sha256'],
            'cache': initial_cache, 'packed_probe_finite': True, 'probes': initial_probes}
        del identity_hidden, probe
        ordered = schedule()
        report['schedule'] = {'seed': 2026092803, 'successful_updates': STEPS,
            'numeric_order_sha256_int64le': runtime.token_digest(torch.tensor(ordered, dtype=torch.int64).numpy()),
            'prose_global_start': 0, 'unchanged_v2_schedule_pair_optimizer_attempt': True}
        print('[setup] fresh master/optimizer/scaler checks and 128/512-token packed parity passed', flush=True)
        successful = attempts = overflows = 0
        limit = 1 if args.smoke else STEPS
        while successful < limit:
            index = successful
            pair = pair_for(index, ordered, examples, windows, prose_order)
            before = copy.deepcopy(scaler.state_dict())
            start = time.monotonic()
            result = attempt(bank, teacher, teacher_execution, pair, optimizer, scaler, index)
            after = copy.deepcopy(scaler.state_dict())
            equal_tree(after, expected_scaler_after(before, result['overflow']), 'scaler transition')
            finite_gradients = all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                   for p in bank.parameters())
            if not result['overflow'] and not finite_gradients:
                raise FloatingPointError('Successful unchanged-v2 attempt left nonfinite gradients')
            student_execution.assert_frozen()
            frozen_identity(teacher, teacher_identity)
            attempts += 1; overflows += int(result['overflow'])
            successful += int(not result['overflow'])
            row = {'attempt': attempts, 'update_index': index, 'successful_updates': successful,
                'case_id': pair['id'], 'schedule_entry': pair['schedule_entry'],
                'prose_window': pair['prose_window'], 'prose_start': pair['prose_start'],
                'answer_targets': int(pair['answer_mask'].sum()), 'seconds': time.monotonic()-start,
                'scaler_before': before, 'scaler_after': after, 'loss_scale_before': before['scale'],
                'lr_factor': lr_factor(index), 'learning_rates': [g['lr'] for g in optimizer.param_groups],
                'all_adapter_gradients_finite_after_attempt': finite_gradients,
                'successful_attempt_requires_v2_preclip_and_hidden_gradients_finite': True, **result}
            report['history'].append(row)
            report.update(successful_updates=successful, attempts=attempts, overflows=overflows)
            if overflows > 8:
                raise RuntimeError('Eight-overflow retry budget exceeded')
            if not args.smoke and successful and successful % 384 == 0 and not result['overflow']:
                checkpoint = save_checkpoint(args.out_dir/f'checkpoint_{successful:04d}.pt',
                    bank, optimizer, scaler, binding, successful, attempts)
                report['checkpoints'].append(checkpoint)
            if attempts % 16 == 0 or successful == limit or result['overflow']:
                report['gpu_memory'] = runtime.gpu_memory_receipt()
                write_json(args.out_dir/'report.json', report)
                print(json.dumps({'updates': successful, 'attempts': attempts, 'overflow': result['overflow'],
                    'mk_ce': result['mk_ce'], 'prose_ce': result['prose_ce'], 'loss_scale': result['loss_scale']}), flush=True)
        report['frozen_base_check'] = bank.assert_base_frozen()
        report['frozen_state_calibration_check'] = student_execution.assert_frozen()
        if native.tensor_hash(student_execution.permutations) != binding['table_sha256']:
            raise RuntimeError('Selected table bytes changed during training')
        report['selected_table_bytes_unchanged'] = True
        report['teacher_base_parameters_frozen'] = all(not p.requires_grad and p.grad is None
            and (id(p), p.data_ptr(), p._version) == teacher_identity[n]
            for n, p in teacher.named_parameters())
        if not report['teacher_base_parameters_frozen']:
            raise RuntimeError('Frozen teacher identity, version or gradients changed')
        report['final_scaler'] = scaler.state_dict()
        check_optimizer(optimizer.state_dict(), bank.state_dict(), successful, lr_factor(successful-1))
        adapter_path = args.out_dir/('discarded_smoke_adapter_fp16.pt' if args.smoke else 'adapter_fp16.pt')
        exported = bank.export_fp16(adapter_path, binding=binding)
        actual_adapter = native.read_fp16(adapter_path, expected_binding=binding)
        if actual_adapter['gate_mode'] != 'soft':
            raise RuntimeError('Exported gate mode differs')
        masters = bank.state_dict()
        for name, value in masters.items():
            if not torch.equal(value.half(), actual_adapter['tensors'][name]):
                raise RuntimeError('Final master cast differs from FP16 export')
        nonzero_v = sum(int(torch.count_nonzero(value))
            for name, value in actual_adapter['tensors'].items() if name.endswith('.V_read'))
        need(nonzero_v > 0, 'Training exported an identity adapter with all-zero V_read')
        report['export_nonidentity_check'] = dict(fp16_V_read_nonzero_elements=nonzero_v,
            fp16_V_read_nonzero=True)
        report['export_master_check'] = {'all_master_casts_equal_export': True, 'master_tensors': 224}
        if args.smoke:
            report['smoke_adapter'] = dict(exported, discarded=True, candidate=False)
        else:
            report['adapter'] = exported
            report['final_checkpoint'] = report['checkpoints'][-1]
            checkpoint = torch.load(args.out_dir/'checkpoint_1536.pt', map_location='cpu', weights_only=True)
            equal_tree(checkpoint['masters'], masters, 'final checkpoint masters')
            equal_tree(checkpoint['optimizer'], optimizer.state_dict(), 'final checkpoint optimizer')
            equal_tree(checkpoint['scaler'], scaler.state_dict(), 'final checkpoint scaler')
            report['final_checkpoint_export_check'] = {'all_master_casts_equal_export': True,
                'master_tensors': 224, 'optimizer_exact': True, 'scaler_exact': True,
                'final_checkpoint_sha256': report['final_checkpoint']['sha256']}
        with torch.no_grad():
            for length in (128,512):
                export_probe = windows[0, :length].cuda()[None]
                training_hidden, _ = bank.forward_hidden(export_probe, use_checkpoint=False)
                proof['export'][str(length)] = dict(token_ids=export_probe.cpu(),
                    training_hidden=training_hidden.cpu())
        del training_hidden, export_probe
        student_execution.close(); bank.close()
        export_probes = []
        with torch.no_grad(), native.install_fp16(student, adapter_path, expected_binding=binding):
            with StatePPLQuantV10(student, permutations, layout=LAYOUT) as deployed:
                for length in (128,512):
                    entry = proof['export'][str(length)]
                    export_probe = entry['token_ids'].cuda()
                    deployed_hidden = deployed.backbone(export_probe, reset=True)
                    deploy_cache = deployed.cache_breakdown()
                    finite_packed(deployed, deployed_hidden)
                    entry['packed_hidden'] = deployed_hidden.cpu()
                    if deploy_cache != initial_caches[length]:
                        raise RuntimeError('Export persistent cache differs')
                    export_probes.append(probe_record(export_probe, entry['packed_hidden'],
                        entry['training_hidden'], deploy_cache))
        report['deployed_export_check'] = {'packed_training_forward_bitwise_equal': True,
            'probe_tokens': 128, 'probe_token_sha256': export_probes[0]['probe_token_sha256'],
            'cache': export_probes[0]['cache'], 'cache_unchanged_from_selected_unadapted': True,
            'packed_probe_finite': True, 'probes': export_probes}
        proof['masters'] = masters
        report['probe_evidence'] = save_probe_evidence(args.out_dir/'probe_evidence.pt', proof)
        report['adapter_payload_bytes'] = sum(t.numel()*t.element_size()
            for t in actual_adapter['tensors'].values())
        need(report['adapter_payload_bytes'] == 2308208, 'Unexpected FP16 adapter payload')
        report['backend_final_check'] = check_replay_backend(policy)
        report['frozen_base_check'] = bank_full_check(bank, base_hashes)
        report['final_teacher_base_check'] = full_loaded_check(teacher, base_hashes, teacher_identity)
        teacher_execution.close()
        report['native_restoration'] = dict(student=restored_native(student, student_forwards),
                                           teacher=restored_native(teacher, teacher_forwards))
        need(code_hashes() == report['code_sha256'], 'Training source changed during execution')
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
