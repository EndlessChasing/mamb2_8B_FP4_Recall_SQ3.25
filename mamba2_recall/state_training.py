"""Stateless, differentiable SQ3.25 execution for Resurface training.

The original model remains frozen. Each mixer invocation is one complete
example (or a batch of complete examples), starting with zero convolution and
SSM history. The differentiable scan uses the deployed quantized carry in its
forward pass, with its explicitly documented surrogate backward. Training
workspace is not an inference-cache memory claim.

Keep this context installed through backward, including block-checkpoint
replay::

    with StateQuantTraining(model, permutations) as execution:
        bank = ResurfaceNative(model, trainable=True)
        hidden, gates = bank.forward_hidden(ids, use_checkpoint=True)
        loss = task_loss(hidden, gates)
        loss.backward()
        bank.close()

This wrapper never remembers a token position or a recurrent tensor. A layer
can therefore be replayed in any order by checkpointing and receives exactly
the same zero-initialized computation. The outer mixer call and ``mx.norm``
call stay intact, preserving the Resurface input and post-D norm prehooks.
"""
from __future__ import annotations

import types
import weakref

import torch

from .state_quant import StateQuant, _OWNERS as _EXECUTION_OWNERS
from .state_training_scan import scan_training


class StateQuantTraining:
    """Install stateless SQ3.25 forwards on the frozen pure Mamba2-8B.

    ``permutations`` is a uint8 tensor shaped [56, 8, 128], whose state
    coordinates are ordered as 16 INT8, 64 INT4, then 48 zero-carry entries.
    It is copied into controller-owned storage and stays fixed during training.
    ``chunk_size`` controls training scan workspace/recomputation, not the
    quantization frequency: the carry is quantized after every token.

    Base parameters are checked at construction, installation and clean context
    exit. ``assert_frozen()`` can also be called at an optimizer boundary. The
    adapter is external to the base parameter tree and may remain trainable.
    """

    def __init__(self, model, permutations, *, chunk_size=32):
        if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)
                or chunk_size <= 0):
            raise ValueError("chunk_size must be a positive integer")
        self.model = model
        self.chunk_size = chunk_size
        self._mixers = [layer.mixer for layer in model.backbone.layers]
        # Reuse the inference path's strict source geometry, dtype, frozen-base
        # and native block-orchestration checks; no inference cache is created.
        StateQuant._validate_model(self)
        self.device = self._mixers[0].in_proj.weight.device
        if self.device.type != "cuda":
            raise ValueError("SQ3.25 training requires a CUDA model")
        self._base = dict(model.named_parameters())
        self._base_identity = {
            name: (id(value), value.data_ptr(), value._version)
            for name, value in self._base.items()
        }
        self.permutations = self._validate_permutations(permutations)
        self._permutation_identity = (
            id(self.permutations), self.permutations.data_ptr(),
            self.permutations._version,
        )
        self._originals = []
        self._installed = False

    def _validate_permutations(self, values):
        if not isinstance(values, torch.Tensor) or values.dtype != torch.uint8:
            raise ValueError("SQ3.25 requires uint8 state permutations")
        if tuple(values.shape) != (56, 8, 128):
            raise ValueError("Expected permutations of shape [56,8,128]")
        cpu = values.detach().cpu().contiguous()
        expected = torch.arange(128, dtype=torch.uint8).expand_as(cpu)
        if not torch.equal(cpu.sort(dim=-1).values, expected):
            raise ValueError("Every state group must permute coordinates 0..127")
        return cpu.to(device=self.device, dtype=torch.uint8).contiguous()

    def assert_frozen(self):
        """Check base and table identity/version; full byte hashes are external.

        This inexpensive guard detects ordinary in-place updates, replacement,
        dtype/device changes and accidental base gradients. It does not replace
        a complete before/after base checksum ledger; no such ledger is implied.
        """
        current = dict(self.model.named_parameters())
        if set(current) != set(self._base):
            raise RuntimeError("The original model parameter set changed")
        for name, value in current.items():
            identity = (id(value), value.data_ptr(), value._version)
            if identity != self._base_identity[name]:
                raise RuntimeError(f"Frozen source parameter changed: {name}")
            if (value.dtype != torch.float16 or value.device != self.device
                    or value.requires_grad or value.grad is not None):
                raise RuntimeError(f"Source parameter is no longer frozen FP16: {name}")
        table = self.permutations
        if (not isinstance(table, torch.Tensor) or table.dtype != torch.uint8
                or table.device != self.device or tuple(table.shape) != (56, 8, 128)
                or not table.is_contiguous()
                or (id(table), table.data_ptr(), table._version)
                != self._permutation_identity):
            raise RuntimeError("State permutation table changed during training")
        return True

    def install(self):
        if self._installed:
            raise RuntimeError("StateQuantTraining is already installed")
        owner = _EXECUTION_OWNERS.get(self.model)
        if owner is not None and owner() is not None:
            raise RuntimeError("Another state execution controller owns this model")
        self.assert_frozen()
        _EXECUTION_OWNERS[self.model] = weakref.ref(self)
        try:
            for index, mx in enumerate(self._mixers):
                self._originals.append((mx, mx.forward))

                def forward(module, u, seqlen=None, seq_idx=None,
                            cu_seqlens=None, inference_params=None, _index=index):
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
        # Restore all native methods even if either training or validation fails.
        try:
            if exc_type is None:
                self.assert_frozen()
        finally:
            self.close()
        return False

    def close(self):
        """Restore native forwards; safe after partial installation or errors."""
        for mx, original in reversed(self._originals):
            mx.forward = original
        self._originals.clear()
        self._installed = False
        owner = _EXECUTION_OWNERS.get(self.model)
        if owner is not None and owner() is self:
            del _EXECUTION_OWNERS[self.model]

    def _forward(self, index, mx, u, seqlen, seq_idx, cu_seqlens, inference_params):
        if not self._installed:
            raise RuntimeError("Training controller was closed before forward/replay")
        if any(value is not None for value in
               (seqlen, seq_idx, cu_seqlens, inference_params)):
            raise ValueError("Training requires complete examples with zero initial state; "
                             "native caches, sequence offsets and packed variable lengths "
                             "are unsupported")
        if (not isinstance(u, torch.Tensor) or u.ndim != 3
                or u.dtype != torch.float16 or u.device != self.device
                or u.shape[0] == 0 or u.shape[1] == 0 or u.shape[2] != 4096):
            raise ValueError("Mixer input must be nonempty CUDA FP16 [batch,tokens,4096]")

        # Rank-three whole-example projection deliberately matches prefill even
        # for a one-token example. There is no continuation/decoding mode here.
        batch, length = u.shape[:2]
        projected = mx.in_proj(u)
        z, xbc, dt = torch.split(projected, (8192, 10240, 128), dim=-1)
        xbc = self._zero_history_convolution(mx, xbc)
        x, bm, cm = torch.split(xbc, (8192, 1024, 1024), dim=-1)
        x = x.reshape(batch, length, 128, 64)
        bm = bm.reshape(batch, length, 8, 128)
        cm = cm.reshape(batch, length, 8, 128)
        aa = -torch.exp(mx.A_log.float())
        y = scan_training(x, dt, aa, bm, cm, mx.D, mx.dt_bias,
                          self.permutations[index], mode="sq3p25",
                          chunk_size=self.chunk_size)
        if y.dtype != torch.float16 or tuple(y.shape) != (batch, length, 128, 64):
            raise RuntimeError("Training scan returned an unexpected readout dtype/shape")
        # Keep the actual norm MODULE invocation: Resurface modifies this
        # post-D readout using the outer mixer's matching input prehook.
        y = mx.norm(y.reshape(batch, length, 8192), z)
        return mx.out_proj(y)

    @staticmethod
    def _zero_history_convolution(mx, xbc):
        """Causal whole-example convolution, with implicit zero left padding."""
        from mamba_ssm.modules import mamba2 as native

        raw = xbc.transpose(1, 2)
        weight = mx.conv1d.weight[:, 0, :]
        if native.causal_conv1d_fn is None:
            # Native Conv1d pads both ends by d_conv-1; remove the trailing
            # positions so every returned value depends only on this example.
            result = mx.act(mx.conv1d(raw))[..., :xbc.shape[1]]
        else:
            result = native.causal_conv1d_fn(
                raw, weight, bias=mx.conv1d.bias,
                activation=mx.activation, seq_idx=None,
            )
        return result.transpose(1, 2)
