"""Pinned NVIDIA Mamba-2 8B source loader and exact tokenizer.

Execution uses state-spaces/mamba (Apache-2.0). The Megatron key mapping follows
NVIDIA's public checkpoint layout. Source BF16 weights are cast to FP16 for the
native quality reference; this is not native Megatron BF16 parity.
"""
from __future__ import annotations

import argparse
import enum
import gc
import hashlib
import importlib
import json
import platform
import re
from pathlib import Path

import torch
from torch import nn

SOURCE_CHECKPOINT_SHA256 = "47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb"
TOKENIZER_SHA256 = "5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09"
TOKENIZER_FILENAME = "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
MODEL_CONFIG = {
    "d_model": 4096, "d_intermediate": 0, "n_layer": 56, "vocab_size": 256000,
    "ssm_cfg": {"layer": "Mamba2", "d_state": 128, "d_conv": 4, "expand": 2,
                "headdim": 64, "ngroups": 8, "chunk_size": 128,
                "rmsnorm": True, "norm_before_gate": False,
                "use_mem_eff_path": False},
    "rms_norm": True, "residual_in_fp32": False, "fused_add_norm": False,
    "pad_vocab_size_multiple": 128, "tie_embeddings": False,
}
_VERIFIED_FILES = {}


class _MegatronModelType(enum.Enum):
    # Passive metadata enum from NVIDIA Megatron-LM core_r0.10.0/core/enums.py.
    encoder_or_decoder = 1
    encoder_and_decoder = 2
    retro_encoder = 3
    retro_decoder = 4


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_digest(ids):
    import numpy as np
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def environment_receipt():
    import importlib.metadata
    packages = {}
    for name in ("torch", "mamba-ssm", "numpy", "triton", "datasets", "sentencepiece"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}


def gpu_memory_receipt():
    if not torch.cuda.is_available():
        return None
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "current_allocated_bytes": torch.cuda.memory_allocated(),
            "note": "Native FP16 quality runtime; stored W4 matrices expand to FP16 GPU weights"}


def checkpoint_path(source_dir):
    directory = Path(source_dir)
    if directory.is_file():
        return directory
    for relative in ("model_optim_rng.pt", "release/mp_rank_00/model_optim_rng.pt"):
        candidate = directory / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"NVIDIA checkpoint missing under {directory}")


def normalize_source_key(key):
    """Anchored mapping: decoder.layers.1 must never match layer 10 or 11."""
    special = {"embedding.word_embeddings.weight": "backbone.embedding.weight",
               "decoder.final_norm.weight": "backbone.norm_f.weight",
               "output_layer.weight": "lm_head.weight"}
    if key in special:
        return special[key]
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", key)
    if match:
        return f"backbone.layers.{int(match[1])}.{match[2]}"
    # The loader can inspect an already mapped state without changing its keys.
    if key.startswith("backbone.") or key == "lm_head.weight":
        return key.replace("backbone.embeddings.", "backbone.embedding.", 1)
    raise ValueError(f"Unrecognized model tensor key: {key}")


def normalize_source_state(state):
    result = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unexpected non-tensor model entry {name}: {type(value).__name__}")
        target = normalize_source_key(name)
        if target in result:
            raise ValueError(f"Duplicate normalized tensor: {target}")
        result[target] = value
    return result


def load_source_state(source_dir, expected_sha256=SOURCE_CHECKPOINT_SHA256):
    """Load a pinned official CPU checkpoint without an unrestricted unpickler.

    Megatron stores an argparse namespace, a model-kind enum, and NumPy RNG state.
    Only those inspected passive metadata types are additionally allowed.
    Unexpected pickle globals fail closed. mmap avoids a second disk copy.
    """
    path = checkpoint_path(source_dir)
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    digest = _VERIFIED_FILES.get(cache_key)
    if digest is None:
        digest = sha256_file(path)
        _VERIFIED_FILES[cache_key] = digest
    if digest != expected_sha256:
        raise ValueError(f"Source checkpoint SHA-256 mismatch: {digest}")
    import numpy as np
    reconstruct = importlib.import_module("numpy.core.multiarray")._reconstruct
    allowed = [argparse.Namespace, np.ndarray, np.dtype, type(np.dtype("uint32")),
               (reconstruct, "numpy.core.multiarray._reconstruct"),
               (_MegatronModelType, "megatron.core.enums.ModelType")]
    try:
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as error:
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
        raise RuntimeError(f"Safe checkpoint loading failed; inspect globals {unsafe}") from error
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    result = normalize_source_state(state)
    del checkpoint
    return result


def make_model(config=None, device="meta", dtype=torch.float16):
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    config = json.loads(json.dumps(MODEL_CONFIG if config is None else config))
    model = MambaLMHeadModel(MambaConfig(**config), device=device, dtype=dtype)
    for block in model.backbone.layers:
        block.mixer.use_mem_eff_path = False
        expected_group_size = block.mixer.d_ssm // block.mixer.ngroups
        if block.mixer.norm.group_size != expected_group_size:
            raise RuntimeError("Runtime does not implement grouped gated RMSNorm")
    return model.eval().requires_grad_(False)


def load_source_model(source_dir, device="cuda", dtype=torch.float16,
                      expected_sha256=SOURCE_CHECKPOINT_SHA256):
    model = make_model()
    state = load_source_state(source_dir, expected_sha256=expected_sha256)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.to(device=device, dtype=dtype)
    if model.backbone.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr():
        raise RuntimeError("8B source requires independent embedding and output head")
    model._package_receipt = {"format": "original BF16 checkpoint cast to requested dtype",
                              "source_checkpoint_sha256": expected_sha256,
                              "parameter_count": sum(p.numel() for p in model.parameters())}
    return model.eval().requires_grad_(False)


class SentencePieceTokenizer:
    """NVIDIA GPTSentencePiece token IDs without automatic BOS/EOS insertion."""
    def __init__(self, path):
        import sentencepiece as spm
        path = Path(path)
        if path.is_dir():
            path = path / TOKENIZER_FILENAME
        self.path = path
        self.processor = spm.SentencePieceProcessor(model_file=str(path))
        self.eos_token_id = self.processor.eos_id()
        self.bos_token_id = self.processor.bos_id()
        self.pad_token_id = self.processor.pad_id()
        self.vocab_size = self.processor.vocab_size()
        self.sha256 = sha256_file(path)
        if self.sha256 != TOKENIZER_SHA256:
            raise ValueError("Tokenizer does not match the pinned original NVIDIA SentencePiece model")

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("This protocol never inserts BOS/EOS automatically")
        return self.processor.encode_as_ids(text)

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.eos_token_id, self.bos_token_id, self.pad_token_id)]
        return self.processor.decode_ids(ids)


def make_cache(model, max_seqlen, batch_size=1, state_dtype=torch.float16):
    """Native one-token cache: state is rounded to the requested dtype each step."""
    from mamba_ssm.utils.generation import InferenceParams
    cache = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
    cache.key_value_memory_dict = model.allocate_inference_cache(
        batch_size, max_seqlen, dtype=state_dtype)
    return cache


def cache_bytes(cache):
    return sum(tensor.numel() * tensor.element_size()
               for pair in cache.key_value_memory_dict.values() for tensor in pair)


@torch.inference_mode()
def backbone_tokenwise(model, ids, state_dtype=torch.float16):
    cache = make_cache(model, ids.shape[1], ids.shape[0], state_dtype)
    result = []
    for position in range(ids.shape[1]):
        cache.seqlen_offset = position
        result.append(model.backbone(ids[:, position:position+1], inference_params=cache))
    return torch.cat(result, dim=1), cache


load_model = load_source_model
