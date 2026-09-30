#!/usr/bin/env python3
"""Independent stdlib audit of complete paired, fixed-checkpoint WT2 test PPL.

Reproduces saved token/window identities and score arithmetic on CPU. It binds
recorded GPU tensor hashes and storage receipts to the sealed release ledger;
it does not recompute model logits or reread the large packed weight shards.
There is no PPL threshold, candidate selection, training, or CUDA import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode = True

FORMAT = 'FP4_SQ325_RESURFACE_WT2_TEST_V1'
AUDIT_FORMAT = 'FP4_SQ325_RESURFACE_WT2_TEST_CPU_AUDIT_V1'
ARMS = ('without_resurface', 'resurface')
REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
VALIDATION_TOKEN_SHA = '5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027'
# Dataset/tokenizer metadata recorded before either arm's quality inference.
TEST_TOKEN_SHA = '5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608'
TEST_TEXT_SHA = '696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83'
TEST_TOKENS = 300_964
GH_COMMIT = 'c15ca4f8f7178f7956076e3d124fe19abbb77cd3'
HF_COMMIT = '6ddb462279d2e9bc9fd92c9bb1e14400cfb96176'
TAG = 'v0.1.0-fp4g16-sq325-resurface'
CACHE_BYTES = 28_499_968
STATIC_BYTES = dict(static_basis_bytes=109_952, static_scale_bytes=1_792,
                    static_predictor_bytes=3_673_216)
STATE_BYTES = CACHE_BYTES + sum(STATIC_BYTES.values())
ADAPTER_BYTES = 2_308_208
WEIGHT_BYTES = 16_473_999_360


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    def bad_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    def unique_keys(pairs):
        result = {}
        for name, value in pairs:
            need(name not in result, 'Duplicate JSON key: ' + name)
            result[name] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=bad_constant,
                      object_pairs_hook=unique_keys)


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def file_in(directory, filename):
    need(isinstance(filename, str) and Path(filename).name == filename,
         'Report/token file must be a local basename')
    path = directory / filename
    need(path.is_file() and not path.is_symlink() and path.resolve().parent == directory.resolve(),
         'Report/token path escaped its directory: ' + filename)
    return path


def bound_small_file(bundle, manifest, relative):
    spec = manifest['files'][relative]
    path = bundle / relative
    need(path.is_file() and not path.is_symlink() and
         path.resolve().is_relative_to(bundle.resolve()), 'Sealed file escaped bundle: ' + relative)
    need(type(spec['bytes']) is int and path.stat().st_size == spec['bytes'] and
         sha(path) == spec['sha256'], 'Sealed small-file binding differs: ' + relative)
    return path


def weight_check(receipt, expected):
    need(receipt['complete'] is True and receipt['passed'] is True and
         receipt['actual_content_checked'] is True and receipt['tensors'] == 507 and
         receipt['decoded_tensor_sha256'] == expected and
         receipt['weight_payload_bytes'] == WEIGHT_BYTES,
         'Complete actual-content check of all 507 decoded weights differs')


def backend_check(receipt, policy):
    need(receipt['singleton_config_unchanged'] is True and
         receipt['selected_config'] == policy['selected_config'] and
         receipt.get('best_config') in (None, policy['selected_config']),
         'Pinned runtime backend changed')


def cache_check(cache, tokens):
    expected = dict(mode='sq3p25', scale_mode='stored_scale', int4_clip=1., diagnostic=None,
        is_3p25_candidate=True, batch_size=1, allocated_layers=56,
        conv_fp16_bytes=4_587_520, ssm_payload_bytes=22_020_096,
        ssm_scale_bytes=1_835_008, ssm_total_bytes=23_855_104,
        permutation_bytes=57_344, total_bytes=CACHE_BYTES,
        calibration_workspace_bytes=0, diagnostic_dense_fp32_bytes=0,
        row_bytes=52, tokens_per_layer=[tokens] * 56)
    need(all(cache.get(key) == value for key, value in expected.items()),
         'Persistent packed state/cache geometry or final token position differs')


def descriptor_check(descriptor, layouts):
    expected = dict(format='FP4_LATENT_STATE_STORAGE_V1', layouts=layouts,
        row_bytes=52, payload_bytes=48, scale_bytes=4,
        permutation_storage_bytes=57_344, total_cache_bytes=CACHE_BYTES, **STATIC_BYTES)
    need(all(descriptor.get(key) == value for key, value in expected.items()) and
         len(descriptor['layers']) == 56, 'Complete static/state storage descriptor differs')
    totals = {key: 0 for key in STATIC_BYTES}
    for index, (layout, layer) in enumerate(zip(layouts, descriptor['layers'])):
        old8, old4, oldzero = map(int, layout.split('_'))
        n8, n4, nzero = old8, old4 - 4, oldzero + 4
        need(n8 + n4 + nzero == 128 and n8 + n4 // 2 + 2 + 4 == 52 and
             n8 % 2 == n4 % 2 == 0 and n4 >= 4, 'Invalid frozen tier geometry')
        widths = dict(lo=n8 // 2, hi=n8 // 2, q4=n4 // 2, latent=2)
        expected_layer = dict(layout=layout, layer=index, n8=n8, n4=n4, nzero=nzero,
            row_bytes=52, payload_bytes=48, scale_bytes=4, latent_coefficients=2,
            latent_dtype='torch.float8_e4m3fn', tensor_widths=widths,
            state_shape=[1, 128, 64, 128], state_bytes=128 * 64 * 52,
            conv_shape=[1, 10240, 4], conv_storage_bytes=81_920,
            static_basis_bytes=8 * nzero * 2 * 2, static_scale_bytes=8 * 2 * 2,
            static_predictor_bytes=8 * (n8 + n4 + 2) * nzero * 2)
        need(all(layer.get(key) == value for key, value in expected_layer.items()),
             'Frozen layer storage differs: ' + str(index))
        expected_tensors = {}
        for key, width in widths.items():
            expected_tensors[key] = dict(shape=[1, 128, 64, width],
                dtype='torch.float8_e4m3fn' if key == 'latent' else 'torch.uint8',
                storage_bytes=128 * 64 * width)
        for key in ('s8', 's4'):
            expected_tensors[key] = dict(shape=[1, 128, 64], dtype='torch.float16',
                                         storage_bytes=128 * 64 * 2)
        need(layer['tensors'] == expected_tensors and
             sum(value['storage_bytes'] for value in layer['tensors'].values()) == layer['state_bytes'],
             'Actual cache tensor dtype/shape/bytes differ: ' + str(index))
        for key in totals:
            totals[key] += layer[key]
    need(totals == STATIC_BYTES and CACHE_BYTES + sum(totals.values()) == STATE_BYTES,
         'Complete predictor/basis/scale accounting differs')


def cache_hashes(receipt):
    names = {f'{layer}.{key}' for layer in range(56)
             for key in ('lo', 'hi', 'q4', 'latent', 's8', 's4', 'conv')}
    need(set(receipt) == names and all(is_digest(value) for value in receipt.values()),
         'Complete per-layer cache hash inventory differs')


def audit(args):
    comp = read(args.comparison)
    manifest = read(args.bundle / 'manifest.json')
    need(comp['format'] == FORMAT and comp['complete'] is True and 'error' not in comp,
         'Complete paired WT2 test evaluation required')
    need(comp['protocol_sha256'] == sha(args.protocol) and
         comp['runner_sha256'] == sha(args.runner) and
         comp['bundle_manifest_sha256'] == sha(args.bundle / 'manifest.json'),
         'Protocol/runner/sealed manifest SHA binding differs')
    need(manifest['format'] == 'MAMBA2_FP4_G16_SQ325_RIDGE_RESURFACE_RELEASE_V1' and
         manifest['complete'] is True and comp['release_tag'] == manifest['release_tag'] == TAG and
         manifest['github_repo'] == 'EndlessChasing/mamb2_8B_FP4_Recall_SQ3.25' and
         manifest['hugging_face_repo'] == 'EndlessChasing/Mamb2_8B_FP4_Recall_SQ3.25' and
         comp['github_commit'] == GH_COMMIT and comp['hf_commit'] == HF_COMMIT and
         comp['checkpoint_frozen_before_test'] is True and
         comp['test_used_for_training_or_selection'] is False and
         comp['project_test_text_seen_in_earlier_2p7b_experiments'] is True and
         comp['source_checkpoint_required'] is False,
         'Fixed public checkpoint/no-selection metadata differs')
    need(manifest['persistent_state_side_bytes'] == STATE_BYTES and
         manifest['state_plus_adapter_bytes'] == STATE_BYTES + ADAPTER_BYTES and
         manifest['adapter_fp16_payload_bytes'] == ADAPTER_BYTES and
         manifest['state_config_payload_bytes'] == 3_842_304 and
         manifest['decoded_resident_weight_bytes'] == WEIGHT_BYTES and
         manifest['packed_resident_kernel'] is False, 'Published memory scope differs')
    ledger_path = bound_small_file(args.bundle, manifest, 'weights/conversion_receipt.json')
    weight_manifest_path = bound_small_file(args.bundle, manifest, 'weights/weight_manifest.json')
    loop_path = bound_small_file(args.bundle, manifest, 'scripts/run_fp4_ridge_resurface_v3.py')
    need(comp['released_ppl_loop_sha256'] == sha(loop_path), 'Released frozen PPL loop binding differs')
    bound_small_file(args.bundle, manifest, 'state_config.pt')
    ledger, weights = read(ledger_path), read(weight_manifest_path)
    need(ledger['complete'] is True and weights['complete'] is True and
         weights['conversion_receipt_sha256'] == sha(ledger_path) ==
         manifest['state_provenance']['conversion_receipt_sha256'] and
         weights['format_name'] == 'fp4_g16_e4m3' and weights['decoded_resident_weight_bytes'] == WEIGHT_BYTES,
         'Packed manifest/conversion ledger differs')
    expected = {name: row['decoded_sha256'] for name, row in ledger['tensors'].items()}
    need(len(expected) == 507 and all(is_digest(value) for value in expected.values()) and
         sum(row['numel'] for row in ledger['tensors'].values()) == 8_236_999_680 and
         comp['expected_weight_hashes'] == expected, 'Frozen 507-weight inventory differs')
    weight_check(comp['initial_weight_check'], expected)
    weight_check(comp['final_weight_check'], expected)
    train_path = bound_small_file(args.bundle, manifest, 'evidence/training_report.json')
    train = read(train_path)
    adapter_path = bound_small_file(args.bundle, manifest, 'adapter_fp16.pt')
    adapter = train['adapter']
    tensor_names = {f'layer{layer}.{key}' for layer in range(56)
                    for key in ('V_read', 'g_read', 'router_w', 'router_b')}
    need(train['complete'] is True and train['format'] == 'FP4_RIDGE_RESURFACE_TRAIN_V3' and
         train['mode'] == 'formal' and train['successful_updates'] == 1536 and
         train['binding'] == manifest['adapter_binding'] and
         adapter['payload_bytes'] == ADAPTER_BYTES and adapter['parameters'] == 1_154_104 and
         adapter['roundtrip_bitwise_equal'] is True and adapter['discarded'] is False and
         set(adapter['tensor_sha256']) == tensor_names and
         all(is_digest(value) for value in adapter['tensor_sha256'].values()) and
         adapter['sha256'] == sha(adapter_path) and adapter['bytes'] == adapter_path.stat().st_size and
         comp['expected_adapter'] == adapter, 'Final 224-tensor adapter export binding differs')
    provenance = manifest['state_provenance']
    need(comp['state_provenance'] == provenance and len(provenance['layouts']) == 56,
         'Frozen state provenance/layout differs')
    policy = manifest['backend_policy']
    need(comp['backend_policy'] == policy and policy['applied_before_first_model_load'] is True and
         policy['candidate_used_for_selection'] is False, 'Published backend policy differs')
    backend_check(comp['backend_final_check'], policy)
    need(comp['adapter_removal_reset_and_cache_exact'] is True,
         'Adapter removal hidden/cache restoration was not exact')
    dataset = comp['dataset']
    fixed = dict(dataset='Salesforce/wikitext', configuration='wikitext-2-raw-v1',
                 split='test', revision_argument=REVISION,
                 document_join='two newline characters', tokenizer_sha256=TOKENIZER_SHA,
                 automatic_special_tokens=False)
    need(all(dataset.get(key) == value for key, value in fixed.items()) and
         manifest['tokenizer_sha256'] == TOKENIZER_SHA and dataset['text_sha256'] == TEST_TEXT_SHA and
         dataset['dataset_fingerprint'] == 'a46124b21ac53738',
         'Official pinned test dataset/tokenizer metadata differs')
    tokens = comp['tokens_file']
    need(tokens['file'] == 'tokens.int64le', 'Canonical raw token stream required')
    token_path = file_in(args.comparison.parent, tokens['file'])
    raw = token_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    need(len(raw) >= 129 * 8 and len(raw) % 8 == 0 and tokens['bytes'] == len(raw) and
         tokens['sha256'] == digest == dataset['token_stream_sha256_int64le'] and
         digest == TEST_TOKEN_SHA and digest != VALIDATION_TOKEN_SHA,
         'Complete saved raw test token hash/bytes differs')
    count = len(raw) // 8
    need(type(dataset['total_tokens']) is int and dataset['total_tokens'] == count == TEST_TOKENS and
         all(0 <= value[0] < 256000 for value in struct.iter_unpack('<q', raw)),
         'Raw int64 token population/vocabulary differs')
    identities = []
    for start in range(0, count - 1, 2048):
        targets = min(2048, count - 1 - start)
        identities.append(dict(start=start, target_tokens=targets,
            token_sha256_int64le=hashlib.sha256(raw[start * 8:(start + targets + 1) * 8]).hexdigest()))
    target_count = count - 1
    windowing = dict(targets_per_full_window=2048, logits_chunk_tokens=64,
        state_reset_each_window=True, final_partial_included=True,
        state_requantized_each_token=True, window_count=len(identities), target_tokens=target_count)
    need(comp['windowing'] == windowing and sum(row['target_tokens'] for row in identities) == target_count,
         'Full token stream/final partial window coverage differs')
    need(set(comp['reports']) == set(ARMS) and set(comp['ppl']) == set(ARMS),
         'Exactly the two fixed paired arms required')
    rows = {}
    for arm in ARMS:
        info = comp['reports'][arm]
        need(info['file'] == arm + '.json', 'Canonical arm filename differs')
        path = file_in(args.comparison.parent, info['file'])
        need(sha(path) == info['sha256'], 'Bound arm report SHA differs: ' + arm)
        row = read(path)
        rows[arm] = row
        adapted = arm == 'resurface'
        need(row['format'] == FORMAT and row['arm'] == arm and row['complete'] is True and
             row['ppl_complete'] is True and row['dataset'] == dataset and
             row['adapter_loaded'] is adapted and row['weight_format'] == 'fp4_g16_e4m3' and
             row['table_sha256'] == provenance['table_sha256'] and
             row['layer_layouts'] == provenance['layouts'] and
             row['static_basis_sha256'] == provenance['static_tensor_sha256']['bases'] and
             row['static_scale_sha256'] == provenance['static_tensor_sha256']['scales'] and
             row['static_predictor_sha256'] == provenance['static_tensor_sha256']['predictors'],
             'Arm completeness/population/state bindings differ: ' + arm)
        ppl = row['ppl']
        need(len(ppl['windows']) == len(identities), 'Incomplete window population: ' + arm)
        total = 0.
        for identity, result in zip(identities, ppl['windows']):
            need({key: result[key] for key in identity} == identity and
                 type(result['start']) is int and type(result['target_tokens']) is int and
                 number(result['nll']) and result['nll'] >= 0 and number(result['ppl']) and
                 result['ppl'] == math.exp(result['nll'] / identity['target_tokens']),
                 'Window target coverage/hash/NLL/PPL arithmetic differs: ' + arm)
            total += result['nll']
        need(number(ppl['nll']) and ppl['nll'] == total and
             type(ppl['target_tokens']) is int and ppl['target_tokens'] == target_count and
             number(ppl['ppl']) and ppl['ppl'] == math.exp(total / target_count) and
             info['ppl'] == comp['ppl'][arm] == ppl['ppl'],
             'Aggregate pooled-token NLL/PPL differs: ' + arm)
        probe = row['repeated_reset_probe']
        need(probe['tokens'] == 128 and probe['hidden_and_cache_exact'] is True and
             is_digest(probe['hidden_sha256']) and probe['token_sha256_int64le'] ==
             hashlib.sha256(raw[:128 * 8]).hexdigest(), 'Repeated reset probe differs: ' + arm)
        cache_hashes(probe['cache_tensor_sha256'])
        cache_hashes(row['ppl_end_cache_tensor_sha256'])
        cache_check(probe['cache'], 128)
        for key in ('cache', 'ppl_cache'):
            cache_check(row[key], identities[-1]['target_tokens'])
        need(row['cache'] == row['ppl_cache'], 'Final PPL cache receipt differs: ' + arm)
        for key in ('storage_descriptor_probe', 'storage_descriptor'):
            descriptor_check(row[key], provenance['layouts'])
        weight_check(row['frozen_weight_check'], expected)
        need(row['frozen_source']['identity_version_gradients_unchanged'] is True and
             row['frozen_source']['tensors'] == 507 and
             row['frozen_source']['parameters'] == 8_236_999_680 and
             all(row['native_restoration'].get(key) is True for key in
                 ('complete', 'native_mixer_forwards_restored', 'native_hooks_restored')),
             'Frozen parameter/native-forward restoration differs: ' + arm)
        backend_check(row['backend_policy_check'], policy)
        storage = row['adapter_storage']
        size = ADAPTER_BYTES if adapted else 0
        expected_storage = dict(loaded=adapted, tensors=224 if adapted else 0,
            parameters=1_154_104 if adapted else 0, fp16_payload_bytes=size,
            resident_storage_bytes=size, persistent_buffer_bytes=0,
            additional_recurrent_cache_bytes=0, ema_enabled=False,
            cache_plus_adapter_bytes=CACHE_BYTES + size)
        need(all(storage.get(key) == value for key, value in expected_storage.items()) and
             row['adapter_sha256'] == (adapter['sha256'] if adapted else None),
             'Off/on adapter presence/residency differs: ' + arm)
        if adapted:
            need(storage['adapter_sha256'] == adapter['sha256'] and
                 storage['serialized_file_bytes'] == adapter['bytes'] and
                 storage['tensor_sha256'] == adapter['tensor_sha256'],
                 'All 224 resident adapter hashes differ from final export')
        else:
            need('tensor_sha256' not in storage and 'adapter_sha256' not in storage,
                 'Adapter-free arm unexpectedly contains an adapter')
        cuda_memory = row['cuda_memory']
        need(type(cuda_memory['peak_allocated_bytes']) is int and
             type(cuda_memory['peak_reserved_bytes']) is int and
             cuda_memory['peak_reserved_bytes'] >= cuda_memory['peak_allocated_bytes'] >= WEIGHT_BYTES and
             isinstance(cuda_memory['scope'], str) and 'not encoded model size' in cuda_memory['scope'],
             'Per-arm CUDA peak receipt/accounting scope differs: ' + arm)
    left, right = (rows[arm]['ppl'] for arm in ARMS)
    relative = right['ppl'] / left['ppl'] - 1
    improved = sum(r['nll'] < l['nll'] for l, r in zip(left['windows'], right['windows']))
    need(number(comp['ppl_relative_change']) and comp['ppl_relative_change'] == relative and
         type(comp['matched_windows_improved']) is int and comp['matched_windows_improved'] == improved,
         'Paired descriptive comparison arithmetic differs')
    memory = dict(persistent_state_side_bytes=STATE_BYTES,
                  state_plus_adapter_bytes=STATE_BYTES + ADAPTER_BYTES,
                  decoded_resident_weight_bytes=WEIGHT_BYTES)
    need(comp['memory'] == memory, 'Comparison memory accounting differs')
    return dict(complete=True, passed=True, dataset=dataset,
        token_stream_sha256_int64le=digest, window_count=len(identities),
        target_tokens=target_count, last_window_targets=identities[-1]['target_tokens'],
        all_target_tokens_scored_once=True, final_partial_verified=True,
        all_window_token_hashes_verified=True, pooled_nll_ppl_arithmetic_verified=True,
        matched_off_on_population=True, ppl=comp['ppl'], nll={arm:rows[arm]['ppl']['nll'] for arm in ARMS},
        ppl_relative_change=relative, matched_windows_improved=improved,
        quality_threshold_applied=False, candidate_selection_performed=False,
        recorded_initial_final_and_arm_weight_hashes_match_all_507=True,
        recorded_adapter_hashes_match_final_224_tensor_export=True,
        complete_cache_and_static_counts_verified=True, memory=memory,
        recorded_cuda_memory={arm:rows[arm]['cuda_memory'] for arm in ARMS},
        backend_policy_matches_published_bundle=True,
        adapter_removal_reset_and_cache_exact=True,
        small_bundle_bindings_sha256={name:manifest['files'][name]['sha256'] for name in
            ('weights/conversion_receipt.json', 'weights/weight_manifest.json',
             'state_config.pt', 'adapter_fp16.pt', 'evidence/training_report.json',
             'scripts/run_fp4_ridge_resurface_v3.py')},
        scope='Independent CPU token coverage and score-arithmetic audit plus sealed-ledger bindings '
              'of recorded GPU tensor identities/storage. Packed shard bytes and GPU logits are not '
              'recomputed; dataset text hash/fingerprint are recorded metadata, not an independent '
              'online dataset fetch. Base pretraining contamination/cross-split text duplication unassessed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison', 'bundle', 'protocol', 'runner', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists() and not args.out.is_symlink() and
         not args.out.resolve().is_relative_to(args.bundle.resolve()),
         'Fresh audit output outside sealed bundle required')
    receipt = dict(format=AUDIT_FORMAT, complete=False, passed=False,
                   uses_stdlib_only=True, cuda_initialized=False,
                   input_report_sha256=sha(args.comparison),
                   protocol_sha256=sha(args.protocol), runner_sha256=sha(args.runner),
                   bundle_manifest_sha256=sha(args.bundle / 'manifest.json'),
                   auditor_sha256=sha(__file__))
    try:
        receipt.update(audit(args))
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key:receipt.get(key) for key in
                     ('complete', 'passed', 'ppl', 'target_tokens', 'window_count', 'error')}, indent=2))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
