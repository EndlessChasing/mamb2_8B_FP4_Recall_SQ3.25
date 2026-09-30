"""Stateless fresh-Resurface training on frozen V10 stored-scale32/32/64.

Inherit immutable source/table guards, native zero-history convolution,
controller ownership, checkpoint-safe installation and cleanup. Override only
the complete-example scan call. No recurrent request state is retained here.
"""
from __future__ import annotations
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall.state_training import StateQuantTraining
from state_ppl_training_scan_v11 import scan_training

PROTOCOL_SHA = 'cebe06473806d74726ac006ff6d5dd316f4fdcfb779eb47f5af271ddb1fa04a6'
LAYOUT = '32_32_64'


class StateQuantTrainingV11(StateQuantTraining):
    """Stateless32 INT8/32 INT4/64zero training; source and table stay frozen."""
    def __init__(self, model, permutations, *, chunk_size=32):
        if type(chunk_size) is not int or not 1 <= chunk_size <= 256:
            raise ValueError('chunk_size must be an integer in[1,256]')
        super().__init__(model, permutations, chunk_size=chunk_size)

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
                          self.permutations[index], chunk_size=self.chunk_size)
        if y.dtype != torch.float16 or tuple(y.shape) != (batch, length, 128, 64):
            raise RuntimeError("Training scan returned an unexpected readout dtype/shape")
        # Keep the actual norm MODULE invocation: Resurface modifies this
        # post-D readout using the outer mixer's matching input prehook.
        y = mx.norm(y.reshape(batch, length, 8192), z)
        return mx.out_proj(y)
