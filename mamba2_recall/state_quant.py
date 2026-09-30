"""Packed recurrent-state execution for the frozen native Mamba2-8B runtime.

Weights, convolution, block residuals and Resurface norm hooks are unchanged.
The SSD scan is replaced by an explicitly sequential state_codec scan that
rounds/quantizes the carry after every token, including every prefill token.
Its current-token readout precedes carry quantization, as in native decoding.

Usage::

    with StateQuant(model, mode="sq3p25", permutations=perms) as execution:
        execution.reset(batch_size=1)
        hidden = execution.backbone(prompt_ids)
        hidden = execution.backbone(next_token_ids)  # same packed cache

Calling model.backbone(ids) also works while installed. Native InferenceParams
are deliberately rejected: their allocator creates a redundant dense SSM
cache. This is an inference-only, fixed-shape, single-stream controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import types
import weakref

import torch
import torch.nn.functional as F

from .state_codec import allocate_state, scan


_OWNERS = weakref.WeakKeyDictionary()


@dataclass
class _LayerCache:
    conv: torch.Tensor
    state: object
    tokens: int = 0


class StateQuant:
    """Install sequential S16 or true packed 3.25-bit carry, with native hooks.

    ``permutations`` is uint8[56,8,128]. In packed order the first 16 state
    coordinates use INT8, the next 64 use INT4 and the last 48 have zero carry.
    Only the compact uint8 table is retained; the codec performs its own loads.

    ``collect_stats`` is allowed only for S16. Statistics sum absolute *stored
    FP16 carry* for every token, and are retained as FP64 group totals. They are
    calibration workspace, separately counted from inference cache storage.
    Resetting a request does not discard accumulated calibration statistics.
    """

    def __init__(self, model, mode="s16", permutations=None, collect_stats=False):
        if mode not in ("s16", "sq3p25"):
            raise ValueError("State mode must be explicitly s16 or sq3p25")
        if collect_stats and mode != "s16":
            raise ValueError("Calibration statistics require the S16 control")
        self.model = model
        self.mode = mode
        self.collect_stats = bool(collect_stats)
        self._mixers = [layer.mixer for layer in model.backbone.layers]
        self._validate_model()
        self.device = self._mixers[0].in_proj.weight.device
        if self.device.type != "cuda":
            raise ValueError("The packed state codec requires a CUDA model")
        self.permutations = self._validate_permutations(permutations)
        self._cache = []
        self._sums = [None] * len(self._mixers)
        self._counts = [0] * len(self._mixers)
        self._originals = []
        self._batch_size = None
        self._expected_layer = 0
        self._call_length = None
        self._failed = False
        self._installed = False
        self._inside_backbone = False

    def _validate_model(self):
        backbone = self.model.backbone
        if len(self._mixers) != 56:
            raise ValueError("This runtime requires the pure 56-layer 8B model")
        if backbone.fused_add_norm or backbone.residual_in_fp32:
            raise ValueError("Native unfused FP16 residual orchestration required")
        params = dict(self.model.named_parameters())
        if len(params) != 507 or sum(p.numel() for p in params.values()) != 8236999680:
            raise ValueError("Source must have 507 original tensors / 8,236,999,680 parameters")
        if any(p.dtype != torch.float16 or p.requires_grad or p.grad is not None
               for p in params.values()):
            raise ValueError("Load the frozen FP16 source with no gradients")
        devices = {p.device for p in params.values()}
        if len(devices) != 1:
            raise ValueError("The frozen model must occupy one CUDA device")
        for index, (layer, mx) in enumerate(zip(backbone.layers, self._mixers)):
            if layer.fused_add_norm or layer.residual_in_fp32 or layer.mlp is not None:
                raise ValueError(f"Unsupported block orchestration at layer {index}")
            geometry = (mx.d_model, mx.nheads, mx.headdim, mx.d_state, mx.ngroups,
                        mx.d_conv, mx.d_ssm, mx.d_inner)
            if geometry != (4096, 128, 64, 128, 8, 4, 8192, 8192):
                raise ValueError(f"Unsupported mixer geometry at layer {index}: {geometry}")
            if mx.use_mem_eff_path or not mx.rmsnorm or mx.D_has_hdim:
                raise ValueError("Explicit gated norm and one D per head required")
            if mx.activation not in ("silu", "swish") or mx.dt_limit != (0.0, float("inf")):
                raise ValueError("Unsupported activation or dt clipping")
            if mx.process_group is not None:
                raise ValueError("Distributed tensor parallel execution is unsupported")
            if mx.norm.group_size != 1024:
                raise ValueError("Expected eight gated RMSNorm groups")
            expected = ((18560, 4096), (4096, 8192), (10240, 1, 4))
            actual = tuple(tuple(v.shape) for v in
                           (mx.in_proj.weight, mx.out_proj.weight, mx.conv1d.weight))
            if actual != expected:
                raise ValueError(f"Projection/convolution shapes differ: {actual}")

    def _validate_permutations(self, values):
        if self.mode == "s16":
            if values is not None:
                raise ValueError("S16 control must use original state coordinates")
            return None
        if not isinstance(values, torch.Tensor) or values.dtype != torch.uint8:
            raise ValueError("Packed state requires a uint8 tensor of permutations")
        if tuple(values.shape) != (56, 8, 128):
            raise ValueError("Expected permutations of shape [56,8,128]")
        cpu = values.detach().cpu().contiguous()
        expected = torch.arange(128, dtype=torch.uint8).expand_as(cpu)
        if not torch.equal(cpu.sort(dim=-1).values, expected):
            raise ValueError("Every state group must be a permutation of 0..127")
        return cpu.to(device=self.device, dtype=torch.uint8).contiguous()

    def install(self):
        if self._installed:
            raise RuntimeError("StateQuant controller is already installed")
        owner = _OWNERS.get(self.model)
        if owner is not None and owner() is not None:
            raise RuntimeError("Another StateQuant controller owns this model")
        _OWNERS[self.model] = weakref.ref(self)
        try:
            for index, mx in enumerate(self._mixers):
                self._originals.append((mx, mx.forward))

                def forward(module, u, seqlen=None, seq_idx=None, cu_seqlens=None,
                            inference_params=None, _index=index):
                    return self._forward(_index, module, u, seqlen, seq_idx,
                                         cu_seqlens, inference_params)

                mx.forward = types.MethodType(forward, mx)
            self._installed = True
        except Exception:
            self.close()
            raise
        return self

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        for mx, original in reversed(self._originals):
            mx.forward = original
        self._originals.clear()
        self._installed = False
        owner = _OWNERS.get(self.model)
        if owner is not None and owner() is self:
            del _OWNERS[self.model]
        self.clear()

    def clear(self):
        """Release all request caches, retaining calibration totals and tables."""
        self._cache.clear()
        self._batch_size = None
        self._expected_layer = 0
        self._call_length = None
        self._failed = False

    @torch.no_grad()
    def reset(self, batch_size=1):
        """Allocate fresh zero caches; calibration totals survive request resets."""
        if not self._installed:
            raise RuntimeError("Install the controller before resetting caches")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.clear()
        for mx in self._mixers:
            # Match native channels-last convolution-state strides.
            conv = torch.zeros(batch_size, 4, 10240, device=self.device,
                               dtype=torch.float16).transpose(1, 2)
            state = allocate_state(batch_size, 128, 64, 128, self.mode, self.device)
            self._cache.append(_LayerCache(conv=conv, state=state))
        self._batch_size = batch_size
        return self

    @torch.no_grad()
    def backbone(self, ids, reset=False):
        """Return hidden states, using and advancing this controller's cache."""
        if not self._installed:
            raise RuntimeError("StateQuant controller is not installed")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("Input IDs must be nonempty [batch,tokens]")
        if ids.device != self.device or ids.dtype != torch.long:
            raise ValueError("Input IDs must be CUDA int64 on the model device")
        if self._inside_backbone:
            raise RuntimeError("Concurrent/reentrant backbone invocation is unsupported")
        if reset or not self._cache:
            self.reset(ids.shape[0])
        self._inside_backbone = True
        try:
            result = self.model.backbone(ids)
            if self._expected_layer != 0:
                raise RuntimeError("Backbone did not execute every state-quantized layer")
            return result
        except Exception:
            self._failed = True
            raise
        finally:
            self._inside_backbone = False

    @torch.no_grad()
    def _forward(self, index, mx, u, seqlen, seq_idx, cu_seqlens, inference_params):
        if not self._installed or self._failed:
            raise RuntimeError("Controller is inactive or failed; reset before reusing it")
        if any(value is not None for value in (seqlen, seq_idx, cu_seqlens, inference_params)):
            raise ValueError("Use controller-owned caches; native InferenceParams/varlen are unsupported")
        if (not isinstance(u, torch.Tensor) or u.ndim != 3 or u.dtype != torch.float16
                or u.device != self.device or u.shape[-1] != 4096 or u.shape[1] == 0):
            raise ValueError("Mixer input must be CUDA FP16 [batch,tokens,4096]")
        if not self._cache:
            if index != 0:
                raise RuntimeError("A new request must start with backbone layer zero")
            self.reset(u.shape[0])
        if u.shape[0] != self._batch_size or index != self._expected_layer:
            raise RuntimeError("Batch shape or backbone layer order changed without a reset")
        length = u.shape[1]
        if index == 0:
            if len({entry.tokens for entry in self._cache}) != 1:
                raise RuntimeError("Per-layer cache positions disagree; reset the request")
            self._call_length = length
        elif length != self._call_length:
            raise RuntimeError("Backbone layers received different token counts")
        entry = self._cache[index]
        try:
            is_step = entry.tokens > 0 and length == 1
            # Native step uses rank-two linear/norm inputs; retain those GEMM
            # and adapter-hook shapes as well as its convolution arithmetic.
            projected = mx.in_proj(u.squeeze(1) if is_step else u)
            zxbcdt = projected.unsqueeze(1) if is_step else projected
            z, xbc, dt = torch.split(zxbcdt, (8192, 10240, 128), dim=-1)
            xbc = self._convolution(mx, xbc, entry)
            x, bm, cm = torch.split(xbc, (8192, 1024, 1024), dim=-1)
            batch = u.shape[0]
            x = x.reshape(batch, length, 128, 64)
            bm = bm.reshape(batch, length, 8, 128)
            cm = cm.reshape(batch, length, 8, 128)
            aa = -torch.exp(mx.A_log.float())
            permutation = None if self.permutations is None else self.permutations[index]
            y, stats = scan(x, dt, aa, bm, cm, mx.D, mx.dt_bias, entry.state,
                            permutation=permutation, collect_stats=self.collect_stats)
            if y.dtype != torch.float16 or tuple(y.shape) != (batch, length, 128, 64):
                raise RuntimeError("State codec returned unexpected readout dtype/shape")
            if self.collect_stats:
                self._accumulate_statistics(index, stats, batch, length)
            # Keep this module call intact: the external Resurface norm prehook
            # adds its post-D correction here, using the matching mixer input.
            y = y.reshape(batch, length, 8192)
            if is_step:
                y = mx.norm(y[:, 0], z[:, 0])
                out = mx.out_proj(y).unsqueeze(1)
            else:
                y = mx.norm(y, z)
                out = mx.out_proj(y)
            entry.tokens += length
            self._expected_layer = (index + 1) % len(self._mixers)
            if self._expected_layer == 0:
                self._call_length = None
            return out
        except Exception:
            self._failed = True
            raise

    @staticmethod
    def _convolution(mx, xbc, entry):
        """Preserve native prefill and native one-token convolution arithmetic."""
        from mamba_ssm.modules import mamba2 as native

        length = xbc.shape[1]
        weight = mx.conv1d.weight[:, 0, :]
        if entry.tokens > 0 and length == 1:
            value = xbc[:, 0, :]
            if native.causal_conv1d_update is None:
                entry.conv.copy_(torch.roll(entry.conv, shifts=-1, dims=-1))
                entry.conv[:, :, -1] = value
                value = torch.sum(entry.conv * weight, dim=-1)
                if mx.conv1d.bias is not None:
                    value = value + mx.conv1d.bias
                value = mx.act(value).to(dtype=xbc.dtype)
            else:
                value = native.causal_conv1d_update(
                    value, entry.conv, weight, mx.conv1d.bias, mx.activation)
            return value[:, None, :]

        raw = xbc.transpose(1, 2)
        prefix = 0
        if entry.tokens > 0:
            # Only past raw projection values are prepended. The causal operator
            # then produces exactly the requested continuation positions.
            raw = torch.cat((entry.conv[:, :, 1:], raw), dim=-1)
            prefix = 3
        entry.conv.copy_(F.pad(raw, (4 - raw.shape[-1], 0)))
        if native.causal_conv1d_fn is None:
            result = mx.act(mx.conv1d(raw).transpose(1, 2)[:, :-3])
        else:
            result = native.causal_conv1d_fn(
                raw, weight, bias=mx.conv1d.bias,
                activation=mx.activation, seq_idx=None).transpose(1, 2)
        return result[:, prefix:prefix + length]

    def _accumulate_statistics(self, index, stats, batch, length):
        if (not isinstance(stats, torch.Tensor) or stats.dtype != torch.float32
                or stats.device != self.device or tuple(stats.shape) != (batch, 128, 4, 128)):
            raise RuntimeError("Expected codec calibration sum [batch,128,4,128] FP32")
        # Codec sums over time and the sixteen P coordinates in each tile.
        totals = stats.reshape(batch, 8, 16, 4, 128).double().sum(dim=(0, 2, 3))
        if self._sums[index] is None:
            self._sums[index] = totals
        else:
            self._sums[index].add_(totals)
        self._counts[index] += batch * length * 16 * 64

    def reset_statistics(self):
        self._sums = [None] * len(self._mixers)
        self._counts = [0] * len(self._mixers)

    def statistics(self):
        """Return CPU FP64 sums/means [56,8,128] and sample counts [56]."""
        if not self.collect_stats or any(value is None for value in self._sums):
            raise RuntimeError("Complete S16 calibration is required before collecting statistics")
        sums = torch.stack(self._sums).detach().cpu()
        counts = torch.tensor(self._counts, dtype=torch.int64)
        if bool((counts <= 0).any()):
            raise RuntimeError("Calibration has empty layers")
        return {"sum_abs": sums, "sample_count_per_group": counts,
                "mean_abs": sums / counts[:, None, None],
                "semantics": "abs(FP16 stored carry), each token; average over batch, heads within each group, P and time"}

    def cache_breakdown(self):
        """Actual persistent tensor payload, excluding weights and workspaces."""
        conv = sum(value.conv.numel() * value.conv.element_size() for value in self._cache)
        state = sum(int(value.state.nbytes) for value in self._cache)
        scales = sum(tensor.numel() * tensor.element_size()
                     for entry in self._cache for key, tensor in entry.state.tensors.items()
                     if key in ("s4", "s8"))
        tables = 0 if self.permutations is None else self.permutations.numel() * self.permutations.element_size()
        statistics = sum(value.numel() * value.element_size() for value in self._sums if value is not None)
        return {"mode": self.mode, "batch_size": self._batch_size,
                "allocated_layers": len(self._cache), "conv_fp16_bytes": conv,
                "ssm_payload_bytes": state - scales, "ssm_scale_bytes": scales,
                "ssm_total_bytes": state, "permutation_bytes": tables,
                "total_bytes": conv + state + tables,
                "calibration_workspace_bytes": statistics,
                "tokens_per_layer": [value.tokens for value in self._cache],
                "scope": "Actual persistent request cache plus compact permutation table; weights, activations, registers, allocator reserve and calibration workspace excluded from total_bytes"}

    @property
    def cache_bytes(self):
        return self.cache_breakdown()["total_bytes"]


PackedStateRuntime = StateQuant
