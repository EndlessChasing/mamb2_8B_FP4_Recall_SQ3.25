"""Independent affine INT4 file codec and decoded FP16 quality reference.

This module uses only PyTorch and our pinned NVIDIA/native-runtime loader.
The on-disk matrices are packed; the reference model allocates FP16 weights.
No packed-resident inference implementation or reduced VRAM claim is implied.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import torch
from torch import nn

from . import runtime


FORMAT = "mamba2-independent-affine-w4-v1"
MAGIC = b"M2W4PK01"
PREFIX = struct.Struct("<8sII")
GROUP_SIZE = 128
CLIPPING_FACTORS = (1.00, .99, .98, .97, .96, .95, .94, .92, .90)
PARAMETER_COUNT = 8_236_999_680
TENSOR_COUNT = 507
W4_TENSOR_COUNT = 114
DEFAULT_CHUNK_ROWS = 512
MAX_CHUNK_ELEMENTS = 4_194_304
MIN_FREE_BYTES = 256 * 1024 * 1024
QUANTIZATION = {
    "method": "uniform_affine_centered_minmax_weight_mse",
    "group_size": GROUP_SIZE,
    "clipping_factors": list(CLIPPING_FACTORS),
    "scale_dtype": "float16", "offset_dtype": "float16",
    "code_bits": 4, "code_order": "low_nibble_first",
    "comparison_dtype": "float32", "decoded_dtype": "float16",
    "source_reference_dtype": "float16",
    "training_data_used": False,
}


def _host_check():
    if sys.byteorder != "little":
        raise RuntimeError("This implementation requires a little-endian host; files are explicitly little-endian")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _bytes(tensor):
    _host_check()
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _shape(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("Tensor shape must have between 1 and 8 dimensions")
    if any(type(size) is not int or size <= 0 for size in value):
        raise ValueError("Tensor dimensions must be positive integers")
    return tuple(value)


def is_w4_tensor(name):
    return name in ("backbone.embedding.weight", "lm_head.weight") or bool(
        re.fullmatch(r"backbone\.layers\.([0-9]|[1-4][0-9]|5[0-5])\.mixer\.(in_proj|out_proj)\.weight", name))


def pack_nibbles(codes):
    """Pack input-axis code pairs, with the first code in the low nibble."""
    if codes.dtype != torch.uint8 or codes.ndim != 2:
        raise TypeError("Codes must be a two-dimensional uint8 tensor")
    if bool((codes > 15).any()):
        raise ValueError("Four-bit codes must be in [0, 15]")
    if codes.shape[1] % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def unpack_nibbles(packed, columns):
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise TypeError("Packed codes must be a two-dimensional uint8 tensor")
    if type(columns) is not int or columns <= 0 or packed.shape[1] != (columns + 1) // 2:
        raise ValueError("Packed width does not match the declared column count")
    if columns % 2 and bool((packed[:, -1] & 0xf0).any()):
        raise ValueError("Unused high nibble must be zero")
    result = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=packed.device)
    result[:, 0::2] = packed & 15
    result[:, 1::2] = packed >> 4
    return result[:, :columns]


def decode_groups(scales, offsets, codes, group_size=GROUP_SIZE):
    """The normative decode is FP32 multiply, FP32 add, then FP16 rounding."""
    if group_size != GROUP_SIZE:
        raise ValueError("Format v1 fixes group_size=128")
    if codes.ndim != 2 or codes.dtype != torch.uint8:
        raise TypeError("Codes must be a two-dimensional uint8 tensor")
    rows, columns = codes.shape
    groups = (columns + group_size - 1) // group_size
    if scales.shape != (rows, groups) or offsets.shape != scales.shape:
        raise ValueError("Scale/offset shape mismatch")
    if scales.dtype != torch.float16 or offsets.dtype != torch.float16:
        raise TypeError("Scales and offsets must be actual serialized FP16 values")
    if not bool(torch.isfinite(scales).all() & torch.isfinite(offsets).all()) or bool((scales < 0).any()):
        raise ValueError("Scales must be finite and nonnegative; offsets must be finite")
    if bool((codes > 15).any()):
        raise ValueError("Four-bit codes must be in [0, 15]")
    padded = torch.nn.functional.pad(codes, (0, groups * group_size - columns))
    decoded = (padded.reshape(rows, groups, group_size).float() * scales.float().unsqueeze(-1)
               + offsets.float().unsqueeze(-1)).half().reshape(rows, -1)[:, :columns]
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("Serialized group decodes to nonfinite FP16 weights")
    return decoded.contiguous()


@torch.inference_mode()
def quantize_groups(weights, group_size=GROUP_SIZE):
    """Choose the fixed range candidate with minimum actual FP16 decode MSE.

    Source BF16 is first cast to FP16 to match the declared native reference.
    Ties select the earlier (wider) range. Constant or underflowed-scale groups
    use a zero scale and the FP16-rounded group mean. Padded values are excluded.
    """
    if group_size != GROUP_SIZE:
        raise ValueError("Format v1 fixes group_size=128")
    if weights.ndim != 2 or not weights.is_floating_point() or not weights.numel():
        raise ValueError("Quantization requires a nonempty floating-point matrix")
    if weights.numel() > MAX_CHUNK_ELEMENTS:
        raise ValueError("Chunk exceeds the bounded quantizer workspace; use fewer rows")
    reference = weights.half()
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("Input contains nonfinite or FP16-overflowing weights")
    rows, columns = reference.shape
    groups = (columns + group_size - 1) // group_size
    padded_columns = groups * group_size
    x = torch.nn.functional.pad(reference.float(), (0, padded_columns - columns)).reshape(rows, groups, group_size)
    valid = (torch.arange(padded_columns, device=x.device) < columns).reshape(1, groups, group_size)
    minimum = x.masked_fill(~valid, math.inf).amin(-1)
    maximum = x.masked_fill(~valid, -math.inf).amax(-1)
    midpoint = (minimum + maximum) * .5
    span = maximum - minimum
    mean = x.sum(-1) / valid.sum(-1)
    best_error = torch.full_like(minimum, math.inf)
    best_scale = torch.zeros_like(minimum, dtype=torch.float16)
    best_offset = torch.zeros_like(minimum, dtype=torch.float16)
    best_codes = torch.zeros_like(x, dtype=torch.uint8)
    best_candidate = torch.zeros_like(minimum, dtype=torch.int64)
    reference_error = None
    for index, factor in enumerate(CLIPPING_FACTORS):
        clipped_span = span * factor
        scale = (clipped_span / 15).half()
        offset = (midpoint - clipped_span * .5).half()
        # Dividing by a zero scale is undefined; constant groups have code 0.
        zero_scale = scale == 0
        offset = torch.where(zero_scale, mean.half(), offset)
        denominator = torch.where(zero_scale, torch.ones_like(scale), scale).float()
        codes = ((x - offset.float().unsqueeze(-1)) / denominator.unsqueeze(-1)).round().clamp_(0, 15).to(torch.uint8)
        codes = torch.where(zero_scale.unsqueeze(-1) | ~valid, 0, codes)
        decoded = (codes.float() * scale.float().unsqueeze(-1) + offset.float().unsqueeze(-1)).half().float()
        error = ((decoded - x).square() * valid).sum(-1)
        # A rounded endpoint can overflow even when the source is finite.
        error = torch.where(torch.isfinite(error), error, math.inf)
        if reference_error is None:
            reference_error = error
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
        best_offset = torch.where(better, offset, best_offset)
        best_codes = torch.where(better.unsqueeze(-1), codes, best_codes)
        best_candidate = torch.where(better, index, best_candidate)
    if not bool(torch.isfinite(best_error).all()):
        raise ValueError("No finite representation among the fixed clipping candidates")
    stats = {
        "squared_error": float(best_error.double().sum().item()),
        "weight_count": rows * columns,
        "group_count": rows * groups,
        "candidate_group_counts": torch.bincount(best_candidate.flatten(), minlength=len(CLIPPING_FACTORS)).cpu().tolist(),
        "unclipped_squared_error": float(reference_error.double().sum().item()) if bool(torch.isfinite(reference_error).all()) else None,
    }
    return best_scale, best_offset, best_codes.reshape(rows, padded_columns)[:, :columns].contiguous(), stats


def _header(kind, shape):
    shape = list(_shape(list(shape)))
    header = {"kind": kind, "shape": shape, "byte_order": "little", "decoded_dtype": "float16"}
    if kind == "w4_affine_f16":
        if len(shape) != 2:
            raise ValueError("W4 tensors must be matrices")
        header.update(group_size=GROUP_SIZE, scale_dtype="float16", offset_dtype="float16",
                      code_order="low_nibble_first", payload_layout="row_scales_offsets_codes")
    elif kind == "fp16":
        header["payload_layout"] = "row_major"
    else:
        raise ValueError(f"Unsupported tensor kind: {kind}")
    return header


def _payload_bytes(header):
    shape = header["shape"]
    if header["kind"] == "fp16":
        return math.prod(shape) * 2
    rows, columns = shape
    return rows * (4 * ((columns + GROUP_SIZE - 1) // GROUP_SIZE) + (columns + 1) // 2)


def tensor_file_bytes(kind, shape):
    header = _header(kind, shape)
    return PREFIX.size + len(_json_bytes(header)) + _payload_bytes(header)


def read_tensor_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(PREFIX.size)
        if len(prefix) != PREFIX.size:
            raise ValueError("Truncated tensor prefix")
        magic, version, header_length = PREFIX.unpack(prefix)
        if magic != MAGIC or version != 1 or not 0 < header_length <= 65536:
            raise ValueError("Invalid tensor magic, version or header length")
        encoded = stream.read(header_length)
        if len(encoded) != header_length:
            raise ValueError("Truncated tensor header")
        try:
            header = json.loads(encoded)
            expected = _header(header["kind"], header["shape"])
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("Invalid tensor header") from error
        if header != expected:
            raise ValueError("Tensor header does not match the v1 format")
        offset = PREFIX.size + header_length
        if path.stat().st_size != offset + _payload_bytes(header):
            raise ValueError("Truncated tensor payload or unexpected trailing bytes")
    return header, offset


def _rows_per_chunk(columns, requested):
    if type(requested) is not int or requested < 1:
        raise ValueError("chunk_rows must be a positive integer")
    if columns > MAX_CHUNK_ELEMENTS:
        raise ValueError("A single row exceeds the maximum quantization workspace")
    return min(requested, max(1, MAX_CHUNK_ELEMENTS // columns))


@torch.inference_mode()
def iter_decoded_chunks(path, device="cpu", chunk_rows=DEFAULT_CHUNK_ROWS):
    """Yield (flat offset, FP16 chunk); allocate no complete CPU weight copy."""
    _host_check()
    header, payload_offset = read_tensor_header(path)
    shape = header["shape"]
    with Path(path).open("rb") as stream:
        stream.seek(payload_offset)
        if header["kind"] == "fp16":
            count = math.prod(shape)
            chunk_elements = min(MAX_CHUNK_ELEMENTS, max(1, chunk_rows) * (shape[-1] if len(shape) > 1 else 1))
            for start in range(0, count, chunk_elements):
                length = min(chunk_elements, count - start)
                data = bytearray(stream.read(length * 2))
                if len(data) != length * 2:
                    raise ValueError("Tensor file changed or was truncated while reading")
                decoded = torch.frombuffer(data, dtype=torch.float16).to(device=device)
                if not bool(torch.isfinite(decoded).all()):
                    raise ValueError("Nonfinite FP16 tensor")
                yield start, decoded
        else:
            rows, columns = shape
            groups = (columns + GROUP_SIZE - 1) // GROUP_SIZE
            stride = groups * 4 + (columns + 1) // 2
            chunk_rows = _rows_per_chunk(columns, chunk_rows)
            for start in range(0, rows, chunk_rows):
                length = min(chunk_rows, rows - start)
                data = bytearray(stream.read(length * stride))
                if len(data) != length * stride:
                    raise ValueError("Tensor file changed or was truncated while reading")
                raw = torch.frombuffer(data, dtype=torch.uint8).reshape(length, stride)
                scales = raw[:, :groups * 2].contiguous().view(torch.float16).to(device=device)
                offsets = raw[:, groups * 2:groups * 4].contiguous().view(torch.float16).to(device=device)
                packed = raw[:, groups * 4:].contiguous().to(device=device)
                codes = unpack_nibbles(packed, columns)
                decoded = decode_groups(scales, offsets, codes)
                yield start * columns, decoded.flatten()


def decoded_sha256(path, device="cpu", chunk_rows=DEFAULT_CHUNK_ROWS):
    digest = hashlib.sha256()
    for _, chunk in iter_decoded_chunks(path, device=device, chunk_rows=chunk_rows):
        digest.update(_bytes(chunk))
    return digest.hexdigest()


def read_tensor(path, device="cpu", chunk_rows=DEFAULT_CHUNK_ROWS):
    header, _ = read_tensor_header(path)
    result = torch.empty(header["shape"], dtype=torch.float16, device=device)
    flat = result.flatten()
    for start, chunk in iter_decoded_chunks(path, device=device, chunk_rows=chunk_rows):
        flat[start:start + chunk.numel()].copy_(chunk)
    return result


def _space_check(directory, additional_bytes, reserve_bytes=MIN_FREE_BYTES):
    free = shutil.disk_usage(directory).free
    if free < additional_bytes + reserve_bytes:
        raise OSError(f"Insufficient disk space: need {additional_bytes + reserve_bytes:,} bytes including reserve; have {free:,}")


@torch.inference_mode()
def write_tensor(path, source, kind, device="cpu", chunk_rows=DEFAULT_CHUNK_ROWS,
                 reserve_bytes=MIN_FREE_BYTES):
    """Write atomically, then independently read back and hash decoded values."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    header = _header(kind, source.shape)
    encoded = _json_bytes(header)
    expected_bytes = PREFIX.size + len(encoded) + _payload_bytes(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    _space_check(path.parent, expected_bytes, reserve_bytes)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    digest = hashlib.sha256()
    statistics = {"squared_error": 0.0, "weight_count": source.numel(), "group_count": 0,
                  "candidate_group_counts": [0] * len(CLIPPING_FACTORS), "unclipped_squared_error": 0.0}
    try:
        with temporary.open("xb") as stream:
            stream.write(PREFIX.pack(MAGIC, 1, len(encoded)))
            stream.write(encoded)
            if kind == "fp16":
                flat = source.flatten()
                for start in range(0, flat.numel(), MAX_CHUNK_ELEMENTS):
                    decoded = flat[start:start + MAX_CHUNK_ELEMENTS].half()
                    if not bool(torch.isfinite(decoded).all()):
                        raise ValueError("Input contains nonfinite or FP16-overflowing weights")
                    data = _bytes(decoded)
                    stream.write(data)
                    digest.update(data)
            else:
                rows, columns = source.shape
                chunk_rows = _rows_per_chunk(columns, chunk_rows)
                for start in range(0, rows, chunk_rows):
                    weights = source[start:start + chunk_rows].to(device=device)
                    scales, offsets, codes, stats = quantize_groups(weights)
                    packed = pack_nibbles(codes)
                    raw = torch.cat((scales.contiguous().view(torch.uint8),
                                     offsets.contiguous().view(torch.uint8), packed), dim=1)
                    stream.write(_bytes(raw))
                    digest.update(_bytes(decode_groups(scales, offsets, codes)))
                    statistics["squared_error"] += stats["squared_error"]
                    statistics["group_count"] += stats["group_count"]
                    statistics["candidate_group_counts"] = [a + b for a, b in zip(
                        statistics["candidate_group_counts"], stats["candidate_group_counts"])]
                    if statistics["unclipped_squared_error"] is not None:
                        if stats["unclipped_squared_error"] is None:
                            statistics["unclipped_squared_error"] = None
                        else:
                            statistics["unclipped_squared_error"] += stats["unclipped_squared_error"]
                    del weights, scales, offsets, codes, packed, raw
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != expected_bytes:
            raise RuntimeError("Tensor writer produced an unexpected byte count")
        reconstructed_digest = decoded_sha256(temporary, device="cpu", chunk_rows=chunk_rows)
        if reconstructed_digest != digest.hexdigest():
            raise RuntimeError("Independent CPU file readback differs from the writer's FP16 reconstruction")
        file_digest = runtime.sha256_file(temporary)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    entry = {"file": path.name, "kind": kind, "shape": list(source.shape), "numel": source.numel(),
             "file_bytes": expected_bytes, "sha256": file_digest,
             "decoded_sha256": reconstructed_digest}
    if kind == "w4_affine_f16":
        entry.update(group_size=GROUP_SIZE, statistics=statistics)
    return entry


def expected_tensor_shapes():
    model = runtime.make_model(device="meta", dtype=torch.float16)
    return {name: list(value.shape) for name, value in model.state_dict().items()}


def _validate_schema(shapes):
    if len(shapes) != TENSOR_COUNT or sum(math.prod(shape) for shape in shapes.values()) != PARAMETER_COUNT:
        raise ValueError("Native runtime tensor coverage does not match the pinned 507-tensor, 8,236,999,680-parameter model")
    if sum(is_w4_tensor(name) for name in shapes) != W4_TENSOR_COUNT:
        raise ValueError("Native runtime does not expose exactly 114 quantized matrices")


def validate_manifest(manifest, expected_shapes=None, require_complete=True):
    """Fail closed on source/config/algorithm/coverage/file-layout disagreement."""
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT or manifest.get("version") != 1:
        raise ValueError("Unsupported or missing package format")
    if require_complete and manifest.get("complete") is not True:
        raise ValueError("W4 package is incomplete")
    if type(manifest.get("complete")) is not bool:
        raise ValueError("Invalid package completion field")
    if manifest.get("source_checkpoint_sha256") != runtime.SOURCE_CHECKPOINT_SHA256:
        raise ValueError("Manifest does not bind the pinned NVIDIA source")
    if manifest.get("model_config") != runtime.MODEL_CONFIG:
        raise ValueError("Manifest model configuration differs from the pinned native model")
    if manifest.get("quantization") != QUANTIZATION:
        raise ValueError("Manifest quantization differs from frozen format v1")
    if manifest.get("parameter_count") != PARAMETER_COUNT or manifest.get("tensor_count") != TENSOR_COUNT:
        raise ValueError("Manifest model size does not match the pinned source")
    if manifest.get("w4_tensor_count") != W4_TENSOR_COUNT or manifest.get("fp16_tensor_count") != TENSOR_COUNT - W4_TENSOR_COUNT:
        raise ValueError("Manifest tensor-kind coverage is incorrect")
    expected_shapes = expected_tensor_shapes() if expected_shapes is None else expected_shapes
    _validate_schema(expected_shapes)
    entries = manifest.get("tensors")
    if not isinstance(entries, dict):
        raise ValueError("Manifest tensor map is missing")
    if require_complete and set(entries) != set(expected_shapes):
        raise ValueError("Manifest is missing or has unexpected model tensors")
    if not set(entries).issubset(expected_shapes):
        raise ValueError("Manifest has unexpected model tensors")
    filenames = set()
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid tensor entry: {name}")
        shape = _shape(entry.get("shape"))
        kind = "w4_affine_f16" if is_w4_tensor(name) else "fp16"
        if list(shape) != list(expected_shapes[name]) or entry.get("numel") != math.prod(shape) or entry.get("kind") != kind:
            raise ValueError(f"Incorrect shape, size or storage kind: {name}")
        filename = entry.get("file")
        if not isinstance(filename, str) or not re.fullmatch(r"[0-9]{4}\.w4bin", filename) or filename in filenames:
            raise ValueError(f"Unsafe or duplicate tensor filename: {name}")
        filenames.add(filename)
        if entry.get("file_bytes") != tensor_file_bytes(kind, shape):
            raise ValueError(f"Incorrect stored byte count: {name}")
        if kind == "w4_affine_f16" and entry.get("group_size") != GROUP_SIZE:
            raise ValueError(f"Incorrect group size: {name}")
        for field in ("sha256", "decoded_sha256"):
            if not isinstance(entry.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", entry[field]):
                raise ValueError(f"Invalid {field}: {name}")
    if require_complete and manifest.get("tensor_bytes") != sum(entry["file_bytes"] for entry in entries.values()):
        raise ValueError("Manifest tensor byte total is incorrect")
    return manifest


def _entry_path(directory, entry):
    path = Path(directory) / entry["file"]
    if path.is_symlink() or not path.is_file() or path.resolve().parent != Path(directory).resolve():
        raise ValueError(f"Tensor file is missing, linked, or outside package: {entry['file']}")
    return path


def verify_tensor_entry(directory, entry, verify_decoded=True):
    path = _entry_path(directory, entry)
    header, _ = read_tensor_header(path)
    if header != _header(entry["kind"], entry["shape"]) or path.stat().st_size != entry["file_bytes"]:
        raise ValueError(f"Tensor header disagrees with manifest: {path.name}")
    if runtime.sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Tensor file SHA-256 mismatch: {path.name}")
    if verify_decoded and decoded_sha256(path) != entry["decoded_sha256"]:
        raise ValueError(f"Decoded FP16 SHA-256 mismatch: {path.name}")
    return path


def read_manifest(package_dir, expected_shapes=None):
    path = Path(package_dir) / "manifest.json"
    with path.open("rb") as stream:
        manifest = json.load(stream)
    return validate_manifest(manifest, expected_shapes=expected_shapes)


def _save_manifest(path, manifest):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(manifest) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def quantize_package(source_dir, package_dir, device="cuda", chunk_rows=DEFAULT_CHUNK_ROWS,
                     resume=False, progress=None, protocol_path=None):
    """Quantize the pinned mmap source with bounded row chunks and no dense disk copy."""
    _host_check()
    expected_shapes = expected_tensor_shapes()
    _validate_schema(expected_shapes)
    state = runtime.load_source_state(source_dir)
    actual_shapes = {name: list(value.shape) for name, value in state.items()}
    if actual_shapes != expected_shapes:
        raise ValueError("Pinned source state does not strictly match the native model")
    directory = Path(package_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    code_sha256 = {"mamba2_recall/w4.py": runtime.sha256_file(__file__),
                   "mamba2_recall/runtime.py": runtime.sha256_file(runtime.__file__)}
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Package manifest exists; pass resume=True to verify and reuse finished tensors")
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest, expected_shapes, require_complete=False)
        if manifest.get("code_sha256") != code_sha256:
            raise ValueError("Resume source code differs from the package's quantizer/runtime binding")
    else:
        if any(directory.iterdir()):
            raise FileExistsError("New package directory must be empty")
        manifest = {"format": FORMAT, "version": 1, "complete": False,
                    "source_checkpoint_sha256": runtime.SOURCE_CHECKPOINT_SHA256,
                    "license": "Apache-2.0", "source_repo": "nvidia/mamba2-8b-3t-4k",
                    "source_revision": "b915550c63ba9359f88f44d1f6a600d85af27302",
                    "modification": "Independent affine group128 INT4 quantization; small tensors cast FP16",
                    "code_sha256": code_sha256,
                    "model_config": runtime.MODEL_CONFIG, "quantization": QUANTIZATION,
                    "parameter_count": PARAMETER_COUNT, "tensor_count": TENSOR_COUNT,
                    "w4_tensor_count": W4_TENSOR_COUNT, "fp16_tensor_count": TENSOR_COUNT - W4_TENSOR_COUNT,
                    "tensors": {}, "reference_runtime": "packed files decoded to frozen FP16 native weights"}
    if protocol_path is not None:
        protocol_sha256 = runtime.sha256_file(protocol_path)
        if "protocol_sha256" in manifest and manifest["protocol_sha256"] != protocol_sha256:
            raise ValueError("Resume protocol file differs from the package's frozen protocol")
        manifest["protocol_sha256"] = protocol_sha256
    remaining = sum(tensor_file_bytes("w4_affine_f16" if is_w4_tensor(name) else "fp16", shape)
                    for name, shape in expected_shapes.items() if name not in manifest["tensors"])
    _space_check(directory, remaining + 1024 * 1024)
    manifest["complete"] = False
    _save_manifest(manifest_path, manifest)
    for index, name in enumerate(sorted(expected_shapes)):
        if name in manifest["tensors"]:
            verify_tensor_entry(directory, manifest["tensors"][name])
            if progress:
                progress({"tensor": name, "index": index + 1, "total": TENSOR_COUNT, "reused": True})
            continue
        kind = "w4_affine_f16" if is_w4_tensor(name) else "fp16"
        path = directory / f"{index:04d}.w4bin"
        # A crash after renaming the tensor but before updating the manifest can
        # leave a valid orphan. Delete only our exact planned file on resume.
        if path.exists():
            if not resume or path.is_symlink():
                raise FileExistsError(path)
            path.unlink()
        entry = write_tensor(path, state[name], kind, device=device, chunk_rows=chunk_rows)
        manifest["tensors"][name] = entry
        _save_manifest(manifest_path, manifest)
        if progress:
            progress({"tensor": name, "index": index + 1, "total": TENSOR_COUNT,
                      "file_bytes": entry["file_bytes"], "sha256": entry["sha256"],
                      "statistics": entry.get("statistics")})
    manifest["tensor_bytes"] = sum(entry["file_bytes"] for entry in manifest["tensors"].values())
    manifest["complete"] = True
    validate_manifest(manifest, expected_shapes)
    _save_manifest(manifest_path, manifest)
    del state
    return manifest


@torch.no_grad()
def load_w4_model(package_dir, device="cuda"):
    """Strict native FP16 quality reference loaded from the actual packed files."""
    model = runtime.make_model(device="meta", dtype=torch.float16)
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    manifest = read_manifest(package_dir, expected_shapes=expected)
    directory = Path(package_dir)
    for name, entry in manifest["tensors"].items():
        path = verify_tensor_entry(directory, entry, verify_decoded=False)
        tensor = torch.empty(entry["shape"], device=device, dtype=torch.float16)
        flat = tensor.flatten()
        digest = hashlib.sha256()
        for start, chunk in iter_decoded_chunks(path, device=device):
            flat[start:start + chunk.numel()].copy_(chunk)
            digest.update(_bytes(chunk))
        if digest.hexdigest() != entry["decoded_sha256"]:
            raise ValueError(f"Loaded FP16 tensor hash mismatch: {name}")
        module_name, local_name = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        if local_name in module._parameters:
            setattr(module, local_name, nn.Parameter(tensor, requires_grad=False))
        elif local_name in module._buffers:
            module._buffers[local_name] = tensor
        else:
            raise ValueError(f"Native state entry is not a parameter or buffer: {name}")
    if any(value.is_meta for value in model.state_dict().values()):
        raise RuntimeError("Uninitialized meta tensors remain after strict loading")
    if sum(value.numel() for value in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("Loaded model parameter coverage changed")
    if model.backbone.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr():
        raise RuntimeError("The pinned model requires untied embedding and output head")
    model._package_receipt = {
        "format": FORMAT, "manifest_sha256": runtime.sha256_file(directory / "manifest.json"),
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "parameter_count": PARAMETER_COUNT, "tensor_count": TENSOR_COUNT,
        "w4_tensor_count": W4_TENSOR_COUNT, "fp16_tensor_count": TENSOR_COUNT - W4_TENSOR_COUNT,
        "tensor_bytes": manifest["tensor_bytes"],
        "resident_weight_dtype": "float16", "packed_resident_kernel": False,
        "file_hashes_verified": True, "decoded_hashes_verified": True,
    }
    return model.eval().requires_grad_(False)
