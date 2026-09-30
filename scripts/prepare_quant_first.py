#!/usr/bin/env python3
"""Prepare protocol-v2 TRAIN cases and calibrate SQ3.25 before any adapter.

The CPU-only preparation may run separately with --prepare-only. A later full
invocation re-verifies those same cases and writes fresh calibration artifacts;
existing calibration files are never replaced. No DEV/CONFIRM data is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mamba2_recall import resurface_data as data, runtime


PROTOCOL = ROOT / 'docs' / 'QUANT_FIRST_PROTOCOL.md'
PROTOCOL_SHA = '24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb'
PROSE_MANIFEST = ROOT / 'docs' / 'prose_train_manifest.json'
PROSE_MANIFEST_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
TRAIN_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
FORMAT = 'MAMBA2_QUANT_FIRST_CALIBRATION_V1'


def write_new_json(path, value):
    """Exclusive, atomic publication; a partial file is not a valid receipt."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def load_train_tokens(path):
    if data.sha_file(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('The frozen quantize-first protocol changed')
    if data.sha_file(PROSE_MANIFEST) != PROSE_MANIFEST_SHA:
        raise ValueError('The pinned prose TRAIN manifest changed')
    if data.sha_file(path) != TRAIN_SHA:
        raise ValueError('Expected the exact pinned prose TRAIN tensor file')
    manifest = json.loads(PROSE_MANIFEST.read_text())
    train = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(train, torch.Tensor) or train.dtype != torch.int64
            or tuple(train.shape) != (448, 2048)):
        raise ValueError('Expected CPU int64 TRAIN tokens with shape [448,2048]')
    if (manifest.get('complete') is not True or manifest.get('split') != 'train'
            or manifest.get('evaluation_data_used') is not False
            or manifest.get('heldout_used_for_fitting') is not False
            or manifest.get('training_tokens_file_sha256') != TRAIN_SHA
            or manifest.get('source_checkpoint_sha256') != runtime.SOURCE_CHECKPOINT_SHA256
            or manifest.get('tokenizer_sha256') != runtime.TOKENIZER_SHA256
            or runtime.token_digest(train.numpy()) != manifest.get('training_tokens_sha256_int64le')
            or bool((train < 0).any()) or bool((train >= 256000).any())):
        raise ValueError('Pinned TRAIN tensor content/provenance differs')
    return train


def prepare_numeric(data_root, out, tokenizer):
    manifest_path = data_root / 'train' / 'manifest.json'
    if not (data_root / 'train').exists():
        data.prepare_split(data_root, 'train', tokenizer,
                           protocol_path=PROTOCOL, protocol_sha256=PROTOCOL_SHA,
                           tokenizer_sha256=runtime.TOKENIZER_SHA256)
    manifest_sha = data.sha_file(manifest_path)
    manifest, _, examples = data.load_training(data_root, manifest_sha, tokenizer)
    if (manifest['protocol_sha256'] != PROTOCOL_SHA or len(examples) != 1536
            or manifest['row_count'] != 1536):
        raise ValueError('Numeric TRAIN data must bind this protocol and all1536 cases')
    receipt = {'format': 'MAMBA2_QUANT_FIRST_NUMERIC_TRAIN_V1', 'complete': True,
               'protocol_sha256': PROTOCOL_SHA,
               'tokenizer_sha256': tokenizer.sha256,
               'train_manifest_sha256': manifest_sha,
               'row_count': len(examples), 'split': 'train',
               'files': manifest['files'], 'heldout_used': False,
               'source_helper_sha256': data.sha_file(ROOT / 'mamba2_recall' / 'resurface_data.py')}
    receipt_path = out / 'numeric_train_receipt.json'
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != receipt:
            raise ValueError('Existing numeric TRAIN receipt differs')
    else:
        write_new_json(receipt_path, receipt)
    return receipt


@torch.no_grad()
def calibrate(args, train, tokenizer, numeric):
    # Import the CUDA/Triton execution path only for actual calibration.
    from mamba2_recall.state_quant import StateQuant

    started = time.time()
    torch.manual_seed(2026092803)
    torch.cuda.manual_seed_all(2026092803)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = runtime.load_source_model(args.source_dir)
    parameters = dict(model.named_parameters())
    if len(parameters) != 507 or sum(p.numel() for p in parameters.values()) != 8236999680:
        raise ValueError('Wrong unadapted source parameter inventory')
    identities = {name: (id(p), p.data_ptr(), p._version) for name, p in parameters.items()}
    # This function loads the source itself and never imports/installs Resurface.
    if any(mx._forward_pre_hooks or mx._forward_hooks or mx.norm._forward_pre_hooks
           for mx in (layer.mixer for layer in model.backbone.layers)):
        raise RuntimeError('Fresh calibration source unexpectedly has mixer/adapter hooks')
    torch.cuda.reset_peak_memory_stats()
    with StateQuant(model, 's16', collect_stats=True) as execution:
        for index in range(8):
            hidden = execution.backbone(train[index, :512].to('cuda')[None], reset=True)
            if not bool(torch.isfinite(hidden).all()):
                raise FloatingPointError('Nonfinite unadapted calibration hidden states')
            del hidden
            print(f'[no-adapter calibration] {index + 1}/8, {time.time() - started:.1f}s', flush=True)
        stats = execution.statistics()
        workspace = execution.cache_breakdown()
    count = 8 * 512 * 16 * 64
    if (tuple(stats['mean_abs'].shape) != (56, 8, 128)
            or not bool(torch.isfinite(stats['mean_abs']).all())
            or not bool((stats['mean_abs'] >= 0).all())
            or not bool((stats['sample_count_per_group'] == count).all())):
        raise ValueError('Calibration statistics differ from the frozen geometry/counts')
    permutations = torch.argsort(stats['mean_abs'], dim=-1, descending=True,
                                stable=True).to(torch.uint8).contiguous()
    if not torch.equal(permutations.sort(-1).values,
                       torch.arange(128, dtype=torch.uint8).expand(56, 8, 128)):
        raise RuntimeError('Invalid stable calibration permutation')
    for name, p in model.named_parameters():
        if ((id(p), p.data_ptr(), p._version) != identities[name]
                or p.requires_grad or p.grad is not None):
            raise RuntimeError(f'Source parameter changed during calibration: {name}')
    binding = {'protocol_sha256': PROTOCOL_SHA,
               'source_sha256': runtime.SOURCE_CHECKPOINT_SHA256,
               'tokenizer_sha256': tokenizer.sha256,
               'train_file_sha256': TRAIN_SHA,
               'prose_manifest_sha256': PROSE_MANIFEST_SHA,
               'train_manifest_sha256': numeric['train_manifest_sha256'],
               'adapter': None, 'adapter_sha256': None}
    selection = 'first 8 TRAIN rows, first 512 tokens each; reset per row'
    token_hashes = [runtime.token_digest(train[i, :512].numpy()) for i in range(8)]
    payload = {'format': FORMAT, **binding, 'permutations': permutations,
               'statistics': stats, 'selection': selection,
               'train_token_hashes': token_hashes}
    path = args.out / 'calibration.pt'
    pending = path.with_name(path.name + '.pending')
    with pending.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    restored = torch.load(pending, map_location='cpu', weights_only=True)
    if (set(restored) != set(payload) or restored['format'] != FORMAT
            or any(restored[key] != value for key, value in binding.items())
            or not torch.equal(restored['permutations'], permutations)
            or set(restored['statistics']) != set(stats)):
        raise RuntimeError('Restricted calibration serialization roundtrip failed')
    for key, value in stats.items():
        actual = restored['statistics'][key]
        if not (torch.equal(actual, value) if isinstance(value, torch.Tensor) else actual == value):
            raise RuntimeError(f'Statistics serialization changed {key}')
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    pending.replace(path)
    ordered = torch.gather(stats['mean_abs'], -1, permutations.long())
    total = ordered.sum()
    if not bool(total > 0):
        raise FloatingPointError('Calibration has zero total carried-state magnitude')
    mass = {'int8': float(ordered[..., :16].sum() / total),
            'int4': float(ordered[..., 16:80].sum() / total),
            'dead': float(ordered[..., 80:].sum() / total)}
    per_group_total = ordered.sum(-1)
    per_group_dead = torch.where(per_group_total > 0,
                                 ordered[..., 80:].sum(-1) / per_group_total,
                                 torch.zeros_like(per_group_total))
    receipt = {'format': FORMAT, 'complete': True, **binding,
               'file': path.name, 'sha256': data.sha_file(path),
               'bytes': path.stat().st_size, 'fresh_source_no_adapter': True,
               'selection': selection, 'calibration_tokens': 4096, 'heldout_used': False,
               'train_token_hashes': token_hashes, 'token_hashes': token_hashes,
               'sample_count_per_group': stats['sample_count_per_group'].tolist(),
               'permutation_shape': list(permutations.shape),
               'permutation_count': 56 * 8, 'permutation_payload_bytes': permutations.numel(),
               'permutations_sha256_uint8': hashlib.sha256(permutations.numpy().tobytes()).hexdigest(),
               'tier_coordinate_counts': {'int8': 16, 'int4': 64, 'dead': 48},
               'coordinate_abs_mass_fractions': mass,
               'dead_coordinate_abs_mass_fraction': mass['dead'],
               'dead_coordinate_abs_mass_fraction_per_layer_group': per_group_dead.tolist(),
               'statistics_semantics': stats['semantics'],
               'cache_and_workspace': workspace,
               'frozen_source_identity_version_gradients_unchanged': True,
               'serialization_roundtrip_bitwise': True,
               'code_sha256': {str(p.relative_to(ROOT)): data.sha_file(p) for p in
                               [Path(__file__), ROOT / 'mamba2_recall' / 'runtime.py',
                                ROOT / 'mamba2_recall' / 'state_codec.py',
                                ROOT / 'mamba2_recall' / 'state_quant.py']},
               'environment': runtime.environment_receipt(),
               'gpu_memory': runtime.gpu_memory_receipt(),
               'elapsed_seconds': time.time() - started}
    write_new_json(args.out / 'calibration.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--train-tokens', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'training_data' / 'numeric_v2')
    parser.add_argument('--prepare-only', action='store_true',
                        help='CPU-only numeric TRAIN preparation; no model or calibration')
    args = parser.parse_args()
    if args.prepare_only and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES= for --prepare-only')
    if args.data_root.name == 'numeric_v1':
        raise ValueError('Use an isolated numeric_v2 data directory')
    if args.out.is_symlink() or args.data_root.is_symlink():
        raise ValueError('Output/data-root symlinks are unsupported')
    if not args.prepare_only:
        for name in ('calibration.pt', 'calibration.json', 'calibration.pt.pending', 'calibration.json.pending'):
            if (args.out / name).exists() or (args.out / name).is_symlink():
                raise FileExistsError('Fresh calibration output required: ' + str(args.out / name))
    torch.set_num_threads(8)
    train = load_train_tokens(args.train_tokens)
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    numeric = prepare_numeric(args.data_root, args.out, tokenizer)
    summary = {'complete': True, 'prepare_only': args.prepare_only,
               'numeric_train_manifest_sha256': numeric['train_manifest_sha256'],
               'numeric_cases': numeric['row_count'], 'protocol_sha256': PROTOCOL_SHA}
    if args.prepare_only:
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU-only preparation unexpectedly initialized CUDA')
        summary['cuda_initialized'] = False
    else:
        receipt = calibrate(args, train, tokenizer, numeric)
        summary.update(calibration_sha256=receipt['sha256'], calibration_tokens=4096,
                       dead_coordinate_abs_mass_fraction=receipt['dead_coordinate_abs_mass_fraction'])
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
