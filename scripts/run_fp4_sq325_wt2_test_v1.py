#!/usr/bin/env python3
"""Full paired WT2 test PPL from the immutable public model bundle."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/FP4_G16_SQ325_RESURFACE_WT2_TEST_V1_PROTOCOL.md'
FORMAT = 'FP4_SQ325_RESURFACE_WT2_TEST_V1'
ARMS = ('without_resurface', 'resurface')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def put(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() or args.bundle.resolve() in args.out_dir.resolve().parents:
        parser.error('Fresh output outside the sealed bundle required')
    bundle = args.bundle.resolve()
    sys.path.insert(0, str(bundle / 'scripts'))
    sys.path.insert(0, str(bundle))
    from infer_fp4_ridge_resurface_v1 import verify_bundle, read_state, need
    from mamba2_recall import runtime, resurface_native as native
    from mamba2_recall.fp4_checkpoint import load_packed_model
    from mamba2_recall.calibration import load_wikitext_tokens
    from mamba2_recall.evaluation import ppl_windows
    from evaluate_quant_first import FrozenBase
    from evaluate_resurface_more import pin_replay_backend, check_replay_backend
    from run_w4_state_resurface_v1 import native_snapshot, check_native_snapshot, adapter_storage
    from run_fp4_state_quality_v1 import assert_weight_content
    from run_fp4_ridge_resurface_v3 import evaluate_ppl, execution_for
    import run_fp4_latent_state_quality_v1 as latent
    import torch
    need(torch.cuda.is_available(), 'Recorded CUDA backend required')
    manifest = verify_bundle(bundle)
    args.out_dir.mkdir(parents=True)
    started = time.time()
    report = dict(format=FORMAT, complete=False, protocol_sha256=sha(PROTOCOL),
        runner_sha256=sha(__file__), bundle_manifest_sha256=sha(bundle / 'manifest.json'),
        release_tag=manifest['release_tag'], github_commit='c15ca4f8f7178f7956076e3d124fe19abbb77cd3',
        hf_commit='6ddb462279d2e9bc9fd92c9bb1e14400cfb96176',
        checkpoint_frozen_before_test=True, test_used_for_training_or_selection=False,
        project_test_text_seen_in_earlier_2p7b_experiments=True,
        reports={}, source_checkpoint_required=False,
        scope='Complete official WT2 test; fixed published checkpoint; no test-driven modification. '
              'Base pretraining contamination and cross-split text duplicates not audited.')
    path = args.out_dir / 'comparison.json'
    put(path, report)
    try:
        torch.set_num_threads(8)
        torch.manual_seed(20260929)
        torch.cuda.manual_seed_all(20260929)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        policy = pin_replay_backend()
        need(policy == manifest['backend_policy'], 'Backend differs from measured release')
        report['backend_policy'] = policy
        tokenizer = runtime.SentencePieceTokenizer(bundle / 'tokenizer')
        ids, dataset = load_wikitext_tokens(tokenizer, 'test')
        need(dataset['split'] == 'test' and dataset['token_stream_sha256_int64le'] !=
             '5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027', 'Test split identity differs')
        windows = ppl_windows(ids, 2048)
        need(len(ids) > 1 and sum(len(w) - 1 for _, w in windows) == len(ids) - 1,
             'Test targets omitted or repeated')
        raw = ids.numpy().astype('<i8', copy=False).tobytes()
        (args.out_dir / 'tokens.int64le').write_bytes(raw)
        need(hashlib.sha256(raw).hexdigest() == dataset['token_stream_sha256_int64le'], 'Saved tokens differ')
        report.update(dataset=dataset, windowing=dict(targets_per_full_window=2048,
            logits_chunk_tokens=64, state_reset_each_window=True, final_partial_included=True,
            state_requantized_each_token=True, window_count=len(windows), target_tokens=len(ids)-1),
            tokens_file=dict(file='tokens.int64le', bytes=len(raw), sha256=sha(args.out_dir / 'tokens.int64le')))
        put(path, report)
        print(json.dumps(dict(test_dataset=dataset, windows=len(windows), target_tokens=len(ids)-1)), flush=True)
        state = read_state(bundle, manifest)
        ledger = json.loads((bundle / 'weights/conversion_receipt.json').read_text())
        expected = {name: row['decoded_sha256'] for name, row in ledger['tensors'].items()}
        parent = (*state, None, expected, manifest['state_provenance'])
        train = json.loads((bundle / 'evidence/training_report.json').read_text())
        report['expected_weight_hashes'] = expected
        report['expected_adapter'] = train['adapter']
        report['state_provenance'] = manifest['state_provenance']
        report['released_ppl_loop_sha256'] = sha(bundle / 'scripts/run_fp4_ridge_resurface_v3.py')
        model = load_packed_model(bundle / 'weights', device='cuda')
        report['initial_weight_check'] = assert_weight_content(model, expected)
        frozen = FrozenBase(model)
        snapshot = native_snapshot(model)
        rows = {}
        for arm in ARMS:
            adapted = arm == 'resurface'
            manager = (native.install_fp16(model, bundle / 'adapter_fp16.pt',
                        expected_binding=manifest['adapter_binding'], expected_base_hashes=expected)
                       if adapted else contextlib.nullcontext())
            common = dict(format=FORMAT, arm=arm, adapter_loaded=adapted,
                adapter_sha256=train['adapter']['sha256'] if adapted else None,
                dataset=dataset, weight_format='fp4_g16_e4m3',
                table_sha256=manifest['state_provenance']['table_sha256'])
            arm_path = args.out_dir / (arm + '.json')
            torch.cuda.reset_peak_memory_stats()
            with manager as bank:
                row = evaluate_ppl(model, parent, windows, arm_path, common, frozen, policy)
                row['adapter_storage'] = adapter_storage(bank, train['adapter'] if adapted else {}, arm)
            row['native_restoration'] = check_native_snapshot(snapshot)
            row['cuda_memory'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                scope='Peak since before this arm; includes resident FP16 weights, state, activation and scratch. '
                      'Allocator reserve may retain blocks from the preceding arm; not encoded model size.')
            put(arm_path, row)
            rows[arm] = row
            report['reports'][arm] = dict(file=arm_path.name, sha256=sha(arm_path), ppl=row['ppl']['ppl'])
            put(path, report)
        with torch.inference_mode(), execution_for(model, parent) as restored:
            probe = windows[0][1][:128].cuda()[None]
            hidden = restored.backbone(probe, reset=True)
            old = rows['without_resurface']['repeated_reset_probe']
            need(native.tensor_hash(hidden) == old['hidden_sha256'] and
                 latent.cache_identity(restored) == old['cache_tensor_sha256'], 'Adapter removal differs')
        report['adapter_removal_reset_and_cache_exact'] = True
        report['final_weight_check'] = assert_weight_content(model, expected)
        report['backend_final_check'] = check_replay_backend(policy)
        left, right = (rows[a]['ppl'] for a in ARMS)
        need([(x['start'],x['target_tokens'],x['token_sha256_int64le']) for x in left['windows']] ==
             [(x['start'],x['target_tokens'],x['token_sha256_int64le']) for x in right['windows']],
             'Unmatched test populations')
        report.update(complete=True, ppl={a: rows[a]['ppl']['ppl'] for a in ARMS},
            ppl_relative_change=right['ppl']/left['ppl']-1,
            matched_windows_improved=sum(r['nll']<l['nll'] for l,r in zip(left['windows'],right['windows'])),
            memory=dict(persistent_state_side_bytes=manifest['persistent_state_side_bytes'],
                        state_plus_adapter_bytes=manifest['state_plus_adapter_bytes'],
                        decoded_resident_weight_bytes=manifest['decoded_resident_weight_bytes']),
            elapsed_seconds=time.time()-started)
        put(path, report)
        print(json.dumps({k:report[k] for k in ('complete','ppl','ppl_relative_change','matched_windows_improved','elapsed_seconds')},indent=2),flush=True)
    except Exception as error:
        report.update(complete=False,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.time()-started)
        put(path, report)
        raise


if __name__ == '__main__':
    main()
