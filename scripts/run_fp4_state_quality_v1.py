#!/usr/bin/env python3
"""Isolated full S16/SQ3.25 comparison for one frozen weight format per process."""
from __future__ import annotations
import argparse
import contextlib
import copy
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime, w4, resurface_native as native
from mamba2_recall.state_quant import StateQuant
import run_w4_state_repair_v2 as v2
import fp4_state_binding_v1 as binding

need, sha, read_json, save_json = binding.need, binding.sha, binding.read_json, binding.write_json
FORMAT = 'FP4_STATE_QUALITY_EVAL_V1'
code_hashes = binding.code_hashes


def assert_weight_content(model, expected):
    values = model.state_dict()
    need(set(values) == set(expected) and len(values) == 507, 'Actual weight inventory differs')
    actual = {}
    for name, value in values.items():
        need(value.dtype == torch.float16 and not value.is_meta and not value.requires_grad,
             'Loaded weight dtype, allocation or gradient state differs: ' + name)
        actual[name] = native.tensor_hash(value)
        need(actual[name] == expected[name], 'Actual decoded weight bytes changed: ' + name)
    size = sum(value.numel() * value.element_size() for value in values.values())
    need(size == binding.WEIGHT_BYTES, 'Expanded FP16 weight payload differs')
    return dict(complete=True, passed=True, actual_content_checked=True, tensors=507,
                decoded_tensor_sha256=actual, weight_payload_bytes=size)


def validate_fp4_receipt(model, receipt, weight_format, out_dir):
    expected = dict(format='MAMBA2_FP4_WEIGHT_REFERENCE_V1', complete=True,
        format_name=weight_format, protocol_sha256=binding.PROTOCOL_SHA,
        source_checkpoint_sha256=binding.SOURCE_SHA, model_config=runtime.MODEL_CONFIG,
        source_reference_dtype='float16', tensor_count=507, fp4_tensor_count=114, fp16_tensor_count=393,
        parameter_count=binding.PARAMETER_COUNT, resident_weight_bytes=binding.WEIGHT_BYTES,
        packed_resident_kernel=False, serialization='ephemeral_packed_buffers', actual_packed_roundtrip=True,
        logical_encoded_payload_bytes=binding.PAYLOAD_BYTES[weight_format],
        physical_packed_payload_bytes=binding.PAYLOAD_BYTES[weight_format])
    need(all(receipt.get(key) == value for key, value in expected.items()), 'FP4 conversion receipt identity/bytes differ')
    need('file_hashes_verified' not in receipt and 'tensor_bytes' not in receipt,
         'Ephemeral buffers must not claim exported checkpoint file verification or size')
    binding.check_inventory(receipt['code_sha256'])
    need(receipt['code_sha256'].get('mamba2_recall/fp4.py') == sha(ROOT / 'mamba2_recall/fp4.py'),
         'Conversion receipt does not bind the current codec')
    entries = receipt['tensors']; values = model.state_dict()
    need(set(entries) == set(values) and len(entries) == 507, 'Conversion ledger must cover all507 tensors')
    group_size = 64 if weight_format == 'fp4_g64_f16' else 16
    physical = 0; quantized = 0; source_only = 0
    for name, value in values.items():
        entry = entries[name]
        need(entry.get('shape') == list(value.shape) and entry.get('numel') == value.numel(),
             'Conversion ledger shape/count differs: ' + name)
        for key in ('source_fp16_sha256', 'decoded_sha256'):
            need(isinstance(entry.get(key), str) and re.fullmatch('[0-9a-f]{64}', entry[key]),
                 'Conversion tensor hash missing: ' + name + '/' + key)
        if w4.is_w4_tensor(name):
            quantized += 1
            rows, columns = value.shape
            need(columns % group_size == 0, 'Pinned real matrix needs no physical padding')
            groups = rows * (columns // group_size)
            nbytes = dict(codes_bytes=value.numel() // 2,
                scales_bytes=groups * (2 if group_size == 64 else 1),
                global_scale_bytes=0 if group_size == 64 else 4)
            need(all(entry.get(key) == size for key, size in nbytes.items())
                 and entry.get('payload_bytes') == sum(nbytes.values())
                 and entry.get('group_count') == groups and entry.get('packed_roundtrip_bitwise') is True,
                 'Actual FP4 physical payload differs: ' + name)
            counts = entry.get('candidate_group_counts')
            need(isinstance(counts, list) and len(counts) == 11
                 and all(type(count) is int and count >= 0 for count in counts) and sum(counts) == groups,
                 'Fixed multiplier selection counts differ: ' + name)
            for key in ('codes_sha256', 'scales_sha256'):
                need(isinstance(entry.get(key), str) and re.fullmatch('[0-9a-f]{64}', entry[key]),
                     'Actual packed buffer hash missing: ' + name)
            if group_size == 64:
                need(entry.get('global_scale_sha256') is None, 'G64 must have no tensor-scale buffer')
            else:
                need(isinstance(entry.get('global_scale_sha256'), str)
                     and re.fullmatch('[0-9a-f]{64}', entry['global_scale_sha256']), 'G16 tensor-scale hash missing')
            invalid_counts = entry.get('invalid_candidate_group_counts')
            need(isinstance(invalid_counts, list) and len(invalid_counts) == 11
                 and all(type(count) is int and 0 <= count <= groups for count in invalid_counts),
                 'Fixed multiplier invalid-group counts differ: ' + name)
            selected_sse = entry.get('squared_error')
            need(isinstance(selected_sse, (float, int)) and math.isfinite(selected_sse) and selected_sse >= 0,
                 'Selected source-weight SSE must be finite: ' + name)
            unclipped = entry.get('unclipped_squared_error')
            need((unclipped is None and invalid_counts[0] > 0)
                 or (isinstance(unclipped, (float, int)) and math.isfinite(unclipped)
                     and unclipped >= 0 and invalid_counts[0] == 0),
                 'Unclipped SSE must agree with multiplier1 invalid counts: ' + name)
            physical += sum(nbytes.values())
        else:
            source_only += 1
            need(entry['source_fp16_sha256'] == entry['decoded_sha256'], 'Unquantized tensor changed: ' + name)
            physical += value.numel() * 2
    need(quantized == 114 and source_only == 393 and physical == binding.PAYLOAD_BYTES[weight_format],
         'FP4 matrix/nonmatrix coverage or physical total differs')
    evidence = receipt['evidence']
    need(Path(evidence['file']).name == evidence['file'] and evidence.get('samples') == 114,
         'Bounded actual per-matrix evidence missing')
    path = out_dir / evidence['file']
    need(sha(path) == evidence['sha256'] and path.stat().st_size == evidence['bytes'], 'Conversion evidence changed')
    return {name: entry['decoded_sha256'] for name, entry in entries.items()}


def load_weights(args):
    if args.weight_format == 'int4_control':
        need(args.w4_dir is not None, 'INT4 control requires the original packed W4 directory')
        path = args.w4_dir / 'manifest.json'
        need(sha(path) == binding.W4_MANIFEST_SHA, 'Exact INT4 control manifest required')
        manifest = read_json(path)
        model = w4.load_w4_model(args.w4_dir)
        loader = model._package_receipt
        need(loader['manifest_sha256'] == binding.W4_MANIFEST_SHA and loader['file_hashes_verified'] is True
             and loader['decoded_hashes_verified'] is True and loader['source_checkpoint_sha256'] == binding.SOURCE_SHA,
             'INT4 loader did not verify actual packed and decoded contents')
        receipt = dict(format='MAMBA2_INT4_CONTROL_REFERENCE_V1', complete=True, format_name=args.weight_format,
            protocol_sha256=binding.PROTOCOL_SHA, source_checkpoint_sha256=binding.SOURCE_SHA,
            manifest_sha256=binding.W4_MANIFEST_SHA, model_config=runtime.MODEL_CONFIG,
            tensor_count=507, parameter_count=binding.PARAMETER_COUNT, resident_weight_bytes=binding.WEIGHT_BYTES,
            serialization='existing_packed_files', packed_resident_kernel=False,
            weight_loader_receipt=loader, tensors=manifest['tensors'], code_sha256=code_hashes())
        expected = {name: entry['decoded_sha256'] for name, entry in manifest['tensors'].items()}
    else:
        need(args.w4_dir is None, 'FP4 conversion accepts original source weights only')
        from mamba2_recall.fp4 import load_fp4_model
        model = load_fp4_model(args.source_dir, args.weight_format, device='cuda', chunk_rows=args.chunk_rows,
            evidence_path=args.out_dir / 'conversion_evidence.pt',
            progress=lambda row: print('[FP4 materialize] ' + str(row), flush=True))
        receipt = model._fp4_receipt
        expected = validate_fp4_receipt(model, receipt, args.weight_format, args.out_dir)
    need(v2.no_adapter_hooks(model), 'Weight reference has unexpected adapter hooks')
    save_json(args.out_dir / 'conversion_receipt.json', receipt)
    return model, receipt, expected


@contextlib.contextmanager
def s16_execution(model):
    snapshot = v2.native_snapshot(model)
    try:
        with StateQuant(model, 's16') as execution:
            yield execution
    finally:
        v2.check_native_snapshot(snapshot)


def require_s16_cache(execution, tokens):
    actual = execution.cache_breakdown()
    expected = dict(mode='s16', batch_size=1, allocated_layers=56, conv_fp16_bytes=4587520,
        ssm_payload_bytes=117440512, ssm_scale_bytes=0, ssm_total_bytes=117440512,
        permutation_bytes=0, total_bytes=binding.S16_CACHE_BYTES, calibration_workspace_bytes=0,
        tokens_per_layer=[tokens] * 56)
    need(all(actual.get(key) == value for key, value in expected.items()) and execution.permutations is None,
         'S16 physical cache budget/geometry differs')
    return actual


def s16_descriptor(execution):
    layers = []
    for row in execution._cache:
        need(set(row.state.tensors) == {'state'}, 'S16 controller has extra persistent state')
        value = row.state.tensors['state']
        need(list(value.shape) == [1, 128, 64, 128] and value.dtype == torch.float16
             and value.untyped_storage().nbytes() == 2097152
             and list(row.conv.shape) == [1, 10240, 4] and row.conv.dtype == torch.float16
             and row.conv.untyped_storage().nbytes() == 81920, 'S16 physical storage differs')
        layers.append(dict(state_shape=list(value.shape), state_bytes=value.untyped_storage().nbytes(),
            conv_shape=list(row.conv.shape), conv_storage_bytes=row.conv.untyped_storage().nbytes(),
            tensors=dict(state=dict(shape=list(value.shape), dtype=str(value.dtype),
                                   storage_bytes=value.untyped_storage().nbytes()))))
    need(len(layers) == 56 and execution.permutations is None, 'S16 layer/table inventory differs')
    return dict(layout='s16', layers=layers)


def cache_is_finite(execution):
    values = [value for row in execution._cache for value in [row.conv, *row.state.tensors.values()]
              if value.is_floating_point()]
    return bool(torch.stack([torch.isfinite(value).all() for value in values]).all())


@torch.inference_mode()
def s16_probe(execution, window):
    probe = window[:128].cuda()[None]
    first = execution.backbone(probe, reset=True)
    first_sha = native.tensor_hash(first); identity = v2.cache_identity(execution)
    finite = bool(torch.isfinite(first).all()) and cache_is_finite(execution)
    second = execution.backbone(probe, reset=True)
    second_finite = bool(torch.isfinite(second).all()) and cache_is_finite(execution)
    need(first_sha == native.tensor_hash(second) and identity == v2.cache_identity(execution)
         and finite == second_finite, 'Repeated S16 reset output/cache bytes changed')
    if finite:
        need(torch.equal(first, second), 'Repeated finite S16 reset output changed')
    result = dict(tokens=128, hidden_sha256=first_sha, hidden_and_cache_exact=True,
        cache_tensor_sha256=identity, token_sha256_int64le=runtime.token_digest(probe.cpu().numpy()),
        cache=require_s16_cache(execution, 128))
    return result, s16_descriptor(execution), finite


@torch.inference_mode()
def evaluate_s16(model, windows, path, common):
    result = dict(common, candidate_id='s16', candidate_name='s16', diagnostic='s16',
        deployable=False, scale_mode=None, int4_clip=None, layout='s16', variant='s16',
        candidate_table_sha256=None, adapter_loaded=False, adapter_sha256=None,
        complete=False, ppl=dict(windows=[]))
    save_json(path, result); started = time.time(); torch.cuda.reset_peak_memory_stats()
    with s16_execution(model) as execution:
        execution.reset(1)
        result['allocated_cache'] = require_s16_cache(execution, 0)
        result['storage_descriptor_initial'] = s16_descriptor(execution)
        save_json(path, result)
        probe, descriptor, finite = s16_probe(execution, windows[0][1])
        result.update(repeated_reset_probe=probe, storage_descriptor_probe=descriptor, reset_probe_finite=finite)
        save_json(path, result)
        if not finite:
            raise v2.CandidateInvalid('Nonfinite S16 reset probe hidden or persisted cache')
        total = 0.; count = 0
        for index, (start, window) in enumerate(windows):
            tokens = window.cuda(); hidden = execution.backbone(tokens[:-1][None], reset=True)
            if not bool(torch.isfinite(hidden).all()):
                raise v2.CandidateInvalid('Nonfinite S16 PPL hidden')
            v2.finite_cache(execution)
            loss_sum = 0.
            for pos in range(0, hidden.shape[1], 64):
                end = min(pos + 64, hidden.shape[1])
                logits = model.lm_head(hidden[:, pos:end]).float()
                loss = F.cross_entropy(logits.reshape(-1, 256000), tokens[pos+1:end+1], reduction='sum')
                loss_sum += float(loss); del logits, loss
            if not math.isfinite(loss_sum):
                raise v2.CandidateInvalid('Nonfinite S16 PPL NLL')
            targets = len(window)-1; total += loss_sum; count += targets
            result['ppl']['windows'].append(dict(start=start, target_tokens=targets,
                token_sha256_int64le=runtime.token_digest(window.numpy()), nll=loss_sum,
                ppl=v2.finite_exp(loss_sum/targets)))
            result['ppl'].update(nll=total, target_tokens=count, ppl=v2.finite_exp(total/count))
            result['cache'] = require_s16_cache(execution, targets)
            result['storage_descriptor'] = s16_descriptor(execution)
            result['ppl_end_cache_tensor_sha256'] = v2.cache_identity(execution)
            save_json(path, result)
            if index == 0 or (index+1) % 8 == 0 or index+1 == len(windows):
                print(f'[s16 PPL] {index+1}/{len(windows)} ppl={result["ppl"]["ppl"]:.9f}', flush=True)
            del hidden, tokens
        result['ppl_cache'] = copy.deepcopy(result['cache'])
    result.update(complete=True, ppl_complete=True, runtime_table_unchanged=True,
        allocation_storage_validated=True, persistent_float_finite_checks_passed=True,
        controller_restoration=dict(complete=True, native_mixer_forwards_restored=True, native_hooks_restored=True),
        zero_scale_observations=0, gpu_memory=runtime.gpu_memory_receipt(), elapsed_seconds=time.time()-started)
    v2.check_ppl(result, windows)
    return result


def execute_arm(ctx, mode, windows, table, path):
    common = dict(ctx['common'], arm=mode, state_mode=mode)
    try:
        need(v2.no_adapter_hooks(ctx['model']), 'Unexpected adapter hooks')
        if mode == 's16':
            result = evaluate_s16(ctx['model'], windows, path, common)
        else:
            candidate = dict(id='sq325_top16', table_name='frozen_w4_top16', table=table, layouts=binding.LAYOUTS)
            result = v2.evaluate(ctx['model'], candidate, windows, path, common)
    except v2.CandidateInvalid as error:
        result = read_json(path)
        result.update(complete=False, error=str(error), error_type=type(error).__name__,
            invalid_candidate=True, failure_kind='nonfinite_quality', fatal_failure=False)
    except BaseException as error:
        result = read_json(path) if path.exists() else common
        result.update(complete=False, error=repr(error), error_type=type(error).__name__,
            failure_kind='integrity_or_runtime', fatal_failure=True)
        save_json(path, result); raise
    try:
        result['controller_restoration'] = v2.check_native_snapshot(ctx['snapshot'])
        result['frozen_source'] = ctx['frozen'].check()
        result['backend_policy_check'] = v2.check_replay_backend(ctx['policy'])
        need(native.tensor_hash(table) == binding.TABLE_SHA and code_hashes() == ctx['hashes'], 'Table/source changed')
        need(v2.no_adapter_hooks(ctx['model']), 'Adapter hooks appeared')
        result.update(candidate_table_unchanged=True, adapter_hooks_absent=True)
        if result.get('complete'):
            v2.check_ppl(result, windows)
            if mode == 'sq325':
                need(v2.valid_candidate(result), 'Completed SQ3.25 arm failed integrity guards')
        if result.get('gpu_memory') is not None:
            result['gpu_memory']['note'] = 'Native FP16 quality runtime; packed INT4/FP4 reference weights decode to FP16'
    except BaseException as error:
        result.update(complete=False, error=repr(error), error_type=type(error).__name__,
            failure_kind='integrity_or_runtime', fatal_failure=True)
        save_json(path, result); raise
    save_json(path, result)
    return result


def arm_metrics(row):
    if row.get('complete') is not True:
        need(row.get('failure_kind') == 'nonfinite_quality' and row.get('invalid_candidate') is True
             and row.get('fatal_failure') is False, 'Incomplete arm is not an admitted numerical quality failure')
        return dict(complete=False, finite=False, failure_kind='nonfinite_quality',
                    completed_windows=len(row.get('ppl', {}).get('windows', [])))
    return dict(complete=True, finite=True, windows=len(row['ppl']['windows']),
                **{key: row['ppl'][key] for key in ('nll', 'target_tokens', 'ppl')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight-format', choices=binding.WEIGHT_FORMATS, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--w4-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--chunk-rows', type=int, default=512)
    parser.add_argument('--state-selection', type=Path,
        default=ROOT / 'artifacts/repair_v5/screen_v1/selected_calibration.pt')
    parser.add_argument('--kernel-checks', type=Path,
        default=ROOT / 'artifacts/fp4_weight_v1/kernel_checks_v1.json')
    parser.add_argument('--kernel-audit', type=Path,
        default=ROOT / 'artifacts/fp4_weight_v1/kernel_audit_v1.json')
    parser.add_argument('--state-kernel-checks', type=Path, default=ROOT / 'artifacts/repair_v2/kernel_checks_v2.json')
    parser.add_argument('--state-kernel-audit', type=Path, default=ROOT / 'artifacts/repair_v2/kernel_audit_v2.json')
    for name in ('control-report', 'control-audit', 'prior-report', 'prior-audit'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    binding.check_protocol()
    need(not args.out_dir.exists() and not args.out_dir.is_symlink(), 'Fresh output directory required')
    need(args.chunk_rows > 0, 'Positive bounded row chunk required')
    codec_proof = binding.validate_codec_admission(args.kernel_checks, args.kernel_audit)
    need(sha(args.state_kernel_checks) == binding.STATE_CHECKS_SHA
         and sha(args.state_kernel_audit) == binding.STATE_AUDIT_SHA, 'Exact unchanged state-kernel admission required')
    state_proof = v2.validate_kernel_inputs(args.state_kernel_checks, args.state_kernel_audit)
    order = binding.validate_order(args.weight_format, args.control_report, args.control_audit,
                                  args.prior_report, args.prior_audit)
    table = binding.load_table(args.state_selection)
    archives = binding.load_archives()
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256 == binding.TOKENIZER_SHA, 'Tokenizer differs')
    ids, dataset = v2.load_wikitext_tokens(tokenizer, 'validation'); windows = v2.ppl_windows(ids, 2048)
    need(len(windows) == 130 and sum(len(window)-1 for _, window in windows) == 264764
         and dataset['token_stream_sha256_int64le'] == binding.VALIDATION_TOKENS_SHA, 'Full validation population differs')
    for archive in archives.values():
        v2.check_ppl(archive, windows)
    torch.set_num_threads(8); torch.manual_seed(20260929); torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest'); policy = v2.pin_replay_backend()
    need(all(archive['backend_policy'] == policy for archive in archives.values()), 'Pinned archived backend differs')
    hashes = code_hashes(); args.out_dir.mkdir(parents=True); started = time.time()
    model, conversion, expected = load_weights(args)
    initial = assert_weight_content(model, expected)
    conversion_ref = binding.file_receipt(args.out_dir / 'conversion_receipt.json')
    inputs = dict(protocol_sha256=binding.PROTOCOL_SHA, source_checkpoint_sha256=binding.SOURCE_SHA,
        tokenizer_sha256=binding.TOKENIZER_SHA, validation_tokens_sha256=binding.VALIDATION_TOKENS_SHA,
        state_selection_sha256=binding.SELECTION_SHA, state_selection_receipt_sha256=binding.SELECTION_RECEIPT_SHA,
        table_sha256=binding.TABLE_SHA, conversion_receipt_sha256=conversion_ref['sha256'],
        state_kernel_checks_sha256=state_proof['kernel_checks_sha256'],
        state_kernel_audit_sha256=state_proof['kernel_audit_sha256'], **codec_proof, **order,
        int4_archive_sha256={mode: item[1] for mode, item in binding.ARCHIVES.items()})
    common = dict(format=FORMAT, stage='full', protocol_sha256=binding.PROTOCOL_SHA,
        weight_format=args.weight_format, source_checkpoint_sha256=binding.SOURCE_SHA,
        input_binding=inputs, dataset=dataset, tokenizer_sha256=tokenizer.sha256,
        backend_policy=policy, code_sha256=hashes, conversion_receipt_sha256=conversion_ref['sha256'],
        weight_loader_receipt={key: value for key, value in conversion.items() if key != 'tensors'},
        environment=runtime.environment_receipt(), adapter_used=False, mk_used=False,
        heldout_used=True, heldout_used_for_selection=False,
        quality_scope='Historically exposed WikiText validation; no untouched-test quality claim')
    ctx = dict(model=model, frozen=v2.FrozenBase(model), snapshot=v2.native_snapshot(model),
               policy=policy, hashes=hashes, common=common)
    rows = {}; archived_replays = {}
    for mode in ('s16', 'sq325'):
        path = args.out_dir / ('full_' + mode + '.json')
        rows[mode] = execute_arm(ctx, mode, windows, table, path)
        if args.weight_format == 'int4_control':
            archived_replays[mode] = v2.exact_replay(archives[mode], rows[mode], archive=True)
            rows[mode]['archived_replay'] = archived_replays[mode]
            save_json(path, rows[mode])
    with s16_execution(model) as execution:
        after, after_storage, after_finite = s16_probe(execution, windows[0][1])
    before = rows['s16']['repeated_reset_probe']
    before_storage = rows['s16']['storage_descriptor_probe']
    need(before == after and before_storage == after_storage and rows['s16']['reset_probe_finite'] == after_finite,
         'S16 probe did not restore exactly after SQ3.25 removal')
    restoration = dict(complete=True, exact=True, finite=after_finite, tokens=128,
        before=before, after=after, before_storage=before_storage, after_storage=after_storage)
    save_json(args.out_dir / 's16_probe_restoration.json', restoration)
    final = assert_weight_content(model, expected)
    native_proof = v2.check_native_snapshot(ctx['snapshot']); frozen = ctx['frozen'].check()
    backend_proof = v2.check_replay_backend(policy)
    need(code_hashes() == hashes and native.tensor_hash(table) == binding.TABLE_SHA
         and v2.no_adapter_hooks(model), 'Final source/table/adapter integrity changed')
    metrics = {mode: arm_metrics(row) for mode, row in rows.items()}
    finite = all(item['finite'] for item in metrics.values())
    target_checks = dict(ppl_strictly_below_8p4=bool(metrics['sq325']['finite'] and metrics['sq325']['ppl'] < binding.TARGET),
        both_full_arms_finite=finite, state_cache_exact=True, all_integrity_checks_passed=True,
        s16_probe_restored_exact=True, no_adapter=True)
    target_pass = all(target_checks.values())
    result = dict(format=binding.COMPARE_FORMAT, complete=True, stage='full', weight_format=args.weight_format,
        protocol_sha256=binding.PROTOCOL_SHA, source_checkpoint_sha256=binding.SOURCE_SHA,
        input_binding=inputs, code_sha256=hashes, dataset=dataset, backend_policy=policy,
        conversion_receipt=conversion_ref, report_sha256={mode: sha(args.out_dir / ('full_' + mode + '.json')) for mode in rows},
        arm_order=['s16', 'sq325'], metrics=metrics, initial_weight_content_check=initial,
        final_weight_content_check=final, s16_probe_restoration=restoration,
        s16_probe_restoration_receipt=binding.file_receipt(args.out_dir / 's16_probe_restoration.json'),
        archived_replays=archived_replays, native_restoration=native_proof, frozen_source=frozen,
        backend_policy_check=backend_proof, all_integrity_checks_passed=True,
        cache_bytes=dict(s16=binding.S16_CACHE_BYTES, sq325=binding.CACHE_BYTES),
        resident_weight_bytes=binding.WEIGHT_BYTES, adapter_used=False, mk_used=False,
        heldout_used=True, heldout_used_for_selection=False, target=binding.TARGET,
        target_checks=target_checks, target_pass=target_pass,
        state_ppl_relative_change=(metrics['sq325']['ppl']/metrics['s16']['ppl']-1) if finite else None,
        outcome=('finite_target_pass' if target_pass else 'finite_target_miss') if finite else 'nonfinite_quality',
        resurface_training_permitted=False, independent_cpu_audit_required=True,
        elapsed_seconds=time.time()-started,
        memory_scope='FP4 physical in-memory payload is not checkpoint file size or resident GPU weight memory')
    save_json(args.out_dir / 'full_comparison.json', result)
    print('FP4 paired comparison: ' + str(dict(weight_format=args.weight_format, metrics=metrics,
          target_pass=target_pass, outcome=result['outcome'])), flush=True)


if __name__ == '__main__':
    main()
