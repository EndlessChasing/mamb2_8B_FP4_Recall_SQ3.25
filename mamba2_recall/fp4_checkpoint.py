"""Exact packed FP4 G16 checkpoint export/reload; custom Mamba runtime.

The on-disk shards use the public safetensors container layout, with E4M3FN
scales stored as their U8 bytes. Decode still creates resident FP16 weights.
Verification is standard-library-only and checks the pinned conversion ledger.
This module does not change the frozen quantizer or claim native FP4 GEMM.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'MAMBA2_FP4_G16_PACKED_CHECKPOINT_V1'
LEDGER_SHA = '55de4189a7e04c9564a48b0fbb5f37428d3d1bb5b6274831d878b41ef4686c9b'
SOURCE_SHA = '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'
PAYLOAD_BYTES = 4_638_460_360
WEIGHT_BYTES = 16_473_999_360
DTYPE_BYTES = {'U8': 1, 'F16': 2, 'F32': 4}
MAX_HEADER_BYTES = 8 * 1024 * 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def source_code_hashes():
    names = ('mamba2_recall/fp4_checkpoint.py', 'scripts/export_fp4_g16_checkpoint_v1.py',
             'mamba2_recall/fp4.py',
             'mamba2_recall/runtime.py', 'docs/FP4_WEIGHT_V1_PROTOCOL.md')
    return {name: sha_file(ROOT / name) for name in names}


def read_ledger(path):
    need(sha_file(path) == LEDGER_SHA, 'The exact audited FP4 G16 ledger is required')
    ledger = read_json(path)
    need(ledger['complete'] is True and ledger['format_name'] == 'fp4_g16_e4m3'
         and ledger['source_checkpoint_sha256'] == SOURCE_SHA
         and ledger['tensor_count'] == 507 and ledger['fp4_tensor_count'] == 114
         and ledger['parameter_count'] == 8_236_999_680
         and ledger['logical_encoded_payload_bytes'] == PAYLOAD_BYTES,
         'Pinned conversion ledger has unexpected production geometry')
    return ledger


def make_plan(ledger, shard_limit_bytes):
    need(type(shard_limit_bytes) is int and shard_limit_bytes >= 1024 * 1024,
         'Shard limit must be an integer of at least one MiB')
    plan = {}
    serial = 0
    for name, item in sorted(ledger['tensors'].items()):
        if item['kind'] == 'fp16':
            plan[name] = {'kind': 'fp16', 'shape': item['shape'],
                          'file': 'weights-retained-fp16.safetensors', 'key': name}
            continue
        need(item['kind'] == 'fp4_g16_e4m3' and len(item['shape']) == 2,
             'Unexpected quantized matrix kind/shape')
        rows, columns = item['shape']
        need(columns % 16 == 0, 'Production FP4 width is not divisible by 16')
        bytes_per_row = columns // 2 + columns // 16
        max_rows = max(1, (shard_limit_bytes - 4) // bytes_per_row)
        # Keep quantization chunk boundaries identical to the measured recipe.
        if max_rows >= 512:
            max_rows = (max_rows // 512) * 512
        serial += 1
        parts = []
        for number, start in enumerate(range(0, rows, max_rows), 1):
            parts.append({'file': f'weights-fp4-{serial:03d}-part-{number:03d}.safetensors',
                          'row_start': start, 'row_stop': min(rows, start + max_rows)})
        plan[name] = {'kind': 'fp4_g16_e4m3', 'shape': item['shape'], 'parts': parts,
                      'global_scale_file': parts[0]['file']}
    return plan


def _spec(dtype, shape):
    return {'dtype': dtype, 'shape': shape}


def _file_specs(plan):
    result = {'weights-retained-fp16.safetensors': {}}
    for name, item in plan.items():
        if item['kind'] == 'fp16':
            result[item['file']][name] = _spec('F16', item['shape'])
            continue
        columns = item['shape'][1]
        for index, part in enumerate(item['parts']):
            rows = part['row_stop'] - part['row_start']
            spec = {'codes': _spec('U8', [rows, columns // 2]),
                    'block_scale_bytes': _spec('U8', [rows, columns // 16])}
            if index == 0:
                spec['global_scale'] = _spec('F32', [])
            result[part['file']] = spec
    return result


def _header(spec):
    offset = 0
    header = {'__metadata__': {'format': FORMAT, 'scale_bytes': 'float8_e4m3fn'}}
    for name, item in sorted(spec.items()):
        count = math.prod(item['shape'])
        size = count * DTYPE_BYTES[item['dtype']]
        header[name] = dict(item, data_offsets=[offset, offset + size])
        offset += size
    raw = json.dumps(header, separators=(',', ':'), allow_nan=False).encode()
    raw += b' ' * ((-len(raw)) % 8)
    return raw, header, offset


def _create_shard(path, spec):
    raw, header, payload_bytes = _header(spec)
    with Path(path).open('xb') as stream:
        stream.write(struct.pack('<Q', len(raw)))
        stream.write(raw)
        stream.truncate(8 + len(raw) + payload_bytes)
    return 8 + len(raw), header


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate safetensors header entry')
        result[key] = value
    return result


def inspect_shard(path, expected_spec):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'Expected regular safetensors shard')
    with path.open('rb') as stream:
        prefix = stream.read(8)
        need(len(prefix) == 8, 'Truncated safetensors header prefix')
        header_size = struct.unpack('<Q', prefix)[0]
        need(2 <= header_size <= MAX_HEADER_BYTES and header_size % 8 == 0,
             'Invalid safetensors header size/alignment')
        raw = stream.read(header_size)
        need(len(raw) == header_size, 'Truncated safetensors header')
    parsed = json.loads(raw, object_pairs_hook=_unique_json)
    expected_raw, expected_header, payload_size = _header(expected_spec)
    need(parsed == expected_header and raw == expected_raw,
         'Shard tensor inventory/header differs from deterministic plan')
    need(path.stat().st_size == 8 + header_size + payload_size,
         'Shard is truncated or has extra bytes')
    return 8 + header_size, parsed


def _hash_region(path, start, size, digest):
    with Path(path).open('rb') as stream:
        stream.seek(start)
        remaining = size
        while remaining:
            data = stream.read(min(8 * 1024 * 1024, remaining))
            need(data, 'Truncated payload region')
            digest.update(data)
            remaining -= len(data)


def verify_checkpoint(directory, code_root=ROOT):
    """Validate container and exact encoded bytes, without torch or CUDA."""
    directory, code_root = Path(directory), Path(code_root)
    need(directory.is_dir() and not directory.is_symlink(), 'Regular checkpoint directory required')
    manifest = read_json(directory / 'weight_manifest.json')
    need(manifest['format'] == FORMAT and manifest['complete'] is True,
         'Not a complete FP4 G16 checkpoint')
    ledger = read_ledger(directory / 'conversion_receipt.json')
    plan = make_plan(ledger, manifest['shard_limit_bytes'])
    need(manifest['layout'] == plan and manifest['logical_weight_payload_bytes'] == PAYLOAD_BYTES
         and manifest['decoded_resident_weight_bytes'] == WEIGHT_BYTES
         and manifest['source_checkpoint_sha256'] == SOURCE_SHA
         and manifest['conversion_receipt_sha256'] == LEDGER_SHA,
         'Manifest differs from pinned checkpoint identities')
    for name, digest in manifest['code_sha256'].items():
        need(sha_file(code_root / name) == digest, 'Checkpoint runtime source differs: ' + name)
    need(set(manifest['code_sha256']) == set(source_code_hashes()),
         'Incomplete runtime code inventory')
    for name, digest in ledger['code_sha256'].items():
        need(sha_file(code_root / name) == digest, 'Original measured FP4 source differs: ' + name)
    specs = _file_specs(plan)
    need(set(manifest['files']) == set(specs), 'Manifest shard inventory differs')
    headers, actual_hashes = {}, {}
    for filename, spec in sorted(specs.items()):
        path = directory / filename
        headers[filename] = inspect_shard(path, spec)
        actual_hashes[filename] = sha_file(path)
        need(manifest['files'][filename] == {'sha256': actual_hashes[filename], 'bytes': path.stat().st_size},
             'Shard file identity differs: ' + filename)
    need(manifest['shard_file_bytes'] == sum((directory / filename).stat().st_size for filename in specs)
         and manifest['format_name'] == 'fp4_g16_e4m3'
         and manifest['packed_resident_kernel'] is False and manifest['custom_runtime'] is True,
         'Stored size/runtime claims differ from actual package')
    for name, item in plan.items():
        expected = ledger['tensors'][name]
        if item['kind'] == 'fp16':
            base, header = headers[item['file']]
            begin, end = header[name]['data_offsets']
            digest = hashlib.sha256()
            _hash_region(directory / item['file'], base + begin, end - begin, digest)
            need(digest.hexdigest() == expected['decoded_sha256'], 'Retained FP16 tensor differs: ' + name)
        else:
            for key, hash_key in (('codes', 'codes_sha256'), ('block_scale_bytes', 'scales_sha256')):
                digest = hashlib.sha256()
                for part in item['parts']:
                    base, header = headers[part['file']]
                    begin, end = header[key]['data_offsets']
                    _hash_region(directory / part['file'], base + begin, end - begin, digest)
                need(digest.hexdigest() == expected[hash_key], 'Packed tensor bytes differ: ' + name + '/' + key)
            base, header = headers[item['global_scale_file']]
            begin, end = header['global_scale']['data_offsets']
            digest = hashlib.sha256()
            _hash_region(directory / item['global_scale_file'], base + begin, end - begin, digest)
            need(digest.hexdigest() == expected['global_scale_sha256'], 'Tensor global scale differs: ' + name)
    inventory = set(specs) | {'conversion_receipt.json', 'weight_manifest.json', 'SHA256SUMS'}
    need(not any(p.is_symlink() for p in directory.rglob('*')), 'Checkpoint contains symlink')
    need({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == inventory,
         'Unexpected files in checkpoint directory')
    for name in ('conversion_receipt.json', 'weight_manifest.json'):
        actual_hashes[name] = sha_file(directory / name)
    expected_sums = ''.join(f'{actual_hashes[name]}  {name}\n' for name in sorted(inventory - {'SHA256SUMS'}))
    need((directory / 'SHA256SUMS').read_text() == expected_sums, 'Checkpoint checksum list differs')
    return manifest, ledger


def export_checkpoint(source_dir, conversion_receipt, output_dir, *, device='cuda',
                      chunk_rows=512, shard_limit_bytes=256 * 1024 * 1024, progress=None):
    """Export real code/scale bytes in bounded chunks; fail on any ledger mismatch."""
    import torch
    from . import fp4, runtime
    ledger = read_ledger(conversion_receipt)
    for name, digest in ledger['code_sha256'].items():
        need(sha_file(ROOT / name) == digest, 'Frozen conversion source changed: ' + name)
    need(chunk_rows == 512, 'Production export must preserve measured 512-row chunks')
    output_dir = Path(output_dir)
    need(not output_dir.exists() and not output_dir.is_symlink(), 'Fresh export directory required')
    state = runtime.load_source_state(source_dir)
    need(set(state) == set(ledger['tensors']) and sum(x.numel() for x in state.values()) == 8_236_999_680,
         'Source tensor inventory differs')
    plan = make_plan(ledger, shard_limit_bytes)
    specs = _file_specs(plan)
    output_dir.mkdir(parents=True)
    headers = {filename: _create_shard(output_dir / filename, spec) for filename, spec in specs.items()}
    for ordinal, (name, item) in enumerate(sorted(plan.items()), 1):
        source = state[name]
        expected = ledger['tensors'][name]
        need(list(source.shape) == item['shape'], 'Source tensor shape differs: ' + name)
        source_digest, decoded_digest = hashlib.sha256(), hashlib.sha256()
        if item['kind'] == 'fp16':
            base, header = headers[item['file']]
            begin, _ = header[name]['data_offsets']
            with (output_dir / item['file']).open('r+b') as stream:
                stream.seek(base + begin)
                flat = source.reshape(-1)
                for start in range(0, flat.numel(), fp4.MAX_CHUNK_ELEMENTS):
                    part = flat[start:start + fp4.MAX_CHUNK_ELEMENTS].half()
                    need(bool(torch.isfinite(part).all()), 'Nonfinite retained source')
                    raw = fp4._bytes(part)
                    stream.write(raw)
                    source_digest.update(raw)
                    decoded_digest.update(raw)
        else:
            columns = item['shape'][1]
            step = fp4._rows(columns, chunk_rows)
            global_cpu = fp4.tensor_global_scale(source, step)
            global_device = global_cpu.to(device=device)
            global_raw = fp4._bytes(global_cpu)
            need(hashlib.sha256(global_raw).hexdigest() == expected['global_scale_sha256'],
                 'Whole-matrix scale differs: ' + name)
            code_digest, scale_digest = hashlib.sha256(), hashlib.sha256()
            for index, part in enumerate(item['parts']):
                base, header = headers[part['file']]
                with (output_dir / part['file']).open('r+b') as stream:
                    if index == 0:
                        stream.seek(base + header['global_scale']['data_offsets'][0])
                        stream.write(global_raw)
                    for start in range(part['row_start'], part['row_stop'], step):
                        stop = min(start + step, part['row_stop'])
                        reference = source[start:stop].half()
                        source_digest.update(fp4._bytes(reference))
                        packet = fp4.quantize_chunk(reference.to(device=device), 'fp4_g16_e4m3', global_device)
                        decoded = fp4.decode_packet(packet)
                        code_raw, scale_raw = fp4._bytes(packet['packed']), fp4._bytes(packet['scales'])
                        code_digest.update(code_raw)
                        scale_digest.update(scale_raw)
                        decoded_digest.update(fp4._bytes(decoded))
                        local_row = start - part['row_start']
                        stream.seek(base + header['codes']['data_offsets'][0] + local_row * (columns // 2))
                        stream.write(code_raw)
                        stream.seek(base + header['block_scale_bytes']['data_offsets'][0] + local_row * (columns // 16))
                        stream.write(scale_raw)
                        del reference, packet, decoded
            need(code_digest.hexdigest() == expected['codes_sha256'] and scale_digest.hexdigest() == expected['scales_sha256'],
                 'Re-exported code/scale bytes differ from audited quantizer: ' + name)
        need(source_digest.hexdigest() == expected['source_fp16_sha256']
             and decoded_digest.hexdigest() == expected['decoded_sha256'],
             'Source/decoded tensor differs from quality ledger: ' + name)
        if progress:
            progress({'tensor': name, 'index': ordinal, 'total': 507,
                      'decoded_sha256': decoded_digest.hexdigest()})
    del state
    (output_dir / 'conversion_receipt.json').write_bytes(Path(conversion_receipt).read_bytes())
    files = {filename: {'sha256': sha_file(output_dir / filename), 'bytes': (output_dir / filename).stat().st_size}
             for filename in sorted(specs)}
    manifest = {'format': FORMAT, 'complete': True, 'source_checkpoint_sha256': SOURCE_SHA,
                'conversion_receipt_sha256': LEDGER_SHA, 'format_name': 'fp4_g16_e4m3',
                'logical_weight_payload_bytes': PAYLOAD_BYTES, 'decoded_resident_weight_bytes': WEIGHT_BYTES,
                'shard_file_bytes': sum(x['bytes'] for x in files.values()),
                'packed_resident_kernel': False, 'custom_runtime': True,
                'shard_limit_bytes': shard_limit_bytes, 'layout': plan, 'files': files,
                'code_sha256': source_code_hashes(),
                'scope': 'Real packed weights and retained FP16 tensors; excludes tokenizer, state configuration and adapter'}
    write_json(output_dir / 'weight_manifest.json', manifest)
    names = sorted([*specs, 'conversion_receipt.json', 'weight_manifest.json'])
    (output_dir / 'SHA256SUMS').write_text(''.join(f'{sha_file(output_dir / filename)}  {filename}\n' for filename in names))
    verify_checkpoint(output_dir)
    return manifest


def _read_region(path, position, count):
    with Path(path).open('rb') as stream:
        stream.seek(position)
        data = stream.read(count)
    need(len(data) == count, 'Truncated checkpoint data')
    return bytearray(data)


def load_packed_model(directory, *, device='cuda', chunk_rows=512, progress=None):
    """Reload without the source checkpoint; decode bounded chunks to resident FP16."""
    manifest, ledger = verify_checkpoint(directory)
    import torch
    from torch import nn
    from . import fp4, runtime
    directory = Path(directory)
    model = runtime.make_model(device='meta', dtype=torch.float16)
    expected = {name: list(x.shape) for name, x in model.state_dict().items()}
    plan = manifest['layout']
    need(expected == {name: item['shape'] for name, item in plan.items()}, 'Native model geometry differs')
    headers = {filename: inspect_shard(directory / filename, spec)
               for filename, spec in _file_specs(plan).items()}
    for ordinal, (name, item) in enumerate(sorted(plan.items()), 1):
        destination = torch.empty(item['shape'], device=device, dtype=torch.float16)
        digest = hashlib.sha256()
        if item['kind'] == 'fp16':
            base, header = headers[item['file']]
            begin, end = header[name]['data_offsets']
            raw = _read_region(directory / item['file'], base + begin, end - begin)
            value = torch.frombuffer(raw, dtype=torch.float16).reshape(item['shape'])
            destination.copy_(value)
            digest.update(bytes(raw))
            del raw, value
        else:
            columns = item['shape'][1]
            step = fp4._rows(columns, chunk_rows)
            base, header = headers[item['global_scale_file']]
            raw = _read_region(directory / item['global_scale_file'], base + header['global_scale']['data_offsets'][0], 4)
            global_scale = torch.frombuffer(raw, dtype=torch.float32).clone().to(device=device).reshape(())
            for part in item['parts']:
                base, header = headers[part['file']]
                for start in range(part['row_start'], part['row_stop'], step):
                    stop = min(start + step, part['row_stop'])
                    rows, local = stop - start, start - part['row_start']
                    code_raw = _read_region(directory / part['file'], base + header['codes']['data_offsets'][0] + local * (columns // 2), rows * (columns // 2))
                    scale_raw = _read_region(directory / part['file'], base + header['block_scale_bytes']['data_offsets'][0] + local * (columns // 16), rows * (columns // 16))
                    packed = torch.frombuffer(code_raw, dtype=torch.uint8).clone().reshape(rows, columns // 2).to(device=device)
                    scales = torch.frombuffer(scale_raw, dtype=torch.uint8).clone().view(torch.float8_e4m3fn).reshape(rows, columns // 16).to(device=device)
                    packet = {'format': 'fp4_g16_e4m3', 'shape': [rows, columns], 'group_size': 16,
                              'packed': packed, 'scales': scales, 'global_scale': global_scale}
                    decoded = fp4.decode_packet(packet)
                    destination[start:stop].copy_(decoded)
                    digest.update(fp4._bytes(decoded))
                    del code_raw, scale_raw, packed, scales, packet, decoded
        need(digest.hexdigest() == ledger['tensors'][name]['decoded_sha256'], 'Reloaded decoded hash differs: ' + name)
        module_name, local_name = name.rsplit('.', 1)
        module = model.get_submodule(module_name)
        if local_name in module._parameters:
            setattr(module, local_name, nn.Parameter(destination, requires_grad=False))
        elif local_name in module._buffers:
            module._buffers[local_name] = destination
        else:
            raise ValueError('Unknown native tensor: ' + name)
        if progress:
            progress({'tensor': name, 'index': ordinal, 'total': 507, 'decoded_sha256': digest.hexdigest()})
    need(not any(x.is_meta for x in model.state_dict().values()), 'Incomplete packed checkpoint reload')
    need(model.backbone.embedding.weight.data_ptr() != model.lm_head.weight.data_ptr(), 'Unexpected tied embeddings')
    model._fp4_receipt = dict(ledger, serialization='safetensors_packed_checkpoint',
                              source_checkpoint_required=False, packed_checkpoint_manifest_sha256=sha_file(directory / 'weight_manifest.json'))
    return model.eval().requires_grad_(False)
