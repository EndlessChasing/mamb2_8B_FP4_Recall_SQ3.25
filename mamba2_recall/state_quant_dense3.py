"""Separate dense Q3.25 controller; frozen StateQuant code remains unchanged."""
from __future__ import annotations

import torch
from .state_quant import StateQuant, _LayerCache
from .state_codec_dense3 import (allocate_state, scan, validate_table,
                                 validate_finite_result, TABLE_KINDS)


class Dense3StateQuant(StateQuant):
    """Use one uint8 permutation OR int8 equalizer table, never both.

    Table payload is56*8*128=57344bytes. Per-row carry is48byte codes+4byte
    scales, so the batch-one cache including convolution remains28499968bytes.
    Parent installation/ownership/reset guards and native convolution are reused.
    """
    def __init__(self, model, table, *, table_kind='permutation'):
        if table_kind not in TABLE_KINDS:
            raise ValueError('Expected permutation or equalizer table')
        self.table_kind = table_kind
        # The parent calls our table validator; it does not allocate a cache.
        super().__init__(model, mode='sq3p25', permutations=table, collect_stats=False)
        self.mode = 'dense3_' + table_kind

    def _validate_permutations(self, values):
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != (56, 8, 128):
            raise ValueError('Expected dense Q3 table[56,8,128]')
        cpu = values.detach().cpu().contiguous()
        validate_table(cpu.reshape(56*8, 128), 56*8, self.table_kind)
        return cpu.to(device=self.device).contiguous()

    @property
    def table(self):
        return self.permutations

    @torch.no_grad()
    def reset(self, batch_size=1):
        if not self._installed:
            raise RuntimeError('Install controller before resetting')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError('batch_size must be positive integer')
        self.clear()
        for _ in self._mixers:
            conv = torch.zeros(batch_size, 4, 10240, device=self.device,
                               dtype=torch.float16).transpose(1, 2)
            state = allocate_state(batch_size, 128, 64, 128, self.device)
            self._cache.append(_LayerCache(conv=conv, state=state))
        self._batch_size = batch_size
        return self

    @torch.no_grad()
    def _forward(self, index, mx, u, seqlen, seq_idx, cu_seqlens, inference_params):
        if not self._installed or self._failed:
            raise RuntimeError('Controller is inactive or failed; reset before reuse')
        if any(v is not None for v in (seqlen, seq_idx, cu_seqlens, inference_params)):
            raise ValueError('Use controller-owned cache; native/variable-length cache unsupported')
        if (not isinstance(u, torch.Tensor) or u.ndim != 3 or u.dtype != torch.float16
                or u.device != self.device or u.shape[-1] != 4096 or u.shape[1] == 0):
            raise ValueError('Mixer input must be CUDA FP16[batch,tokens,4096]')
        if not self._cache:
            if index != 0:
                raise RuntimeError('A request must start at layer zero')
            self.reset(u.shape[0])
        if u.shape[0] != self._batch_size or index != self._expected_layer:
            raise RuntimeError('Batch/layer order changed without reset')
        length = u.shape[1]
        if index == 0:
            if len({v.tokens for v in self._cache}) != 1:
                raise RuntimeError('Layer cache positions disagree')
            self._call_length = length
        elif length != self._call_length:
            raise RuntimeError('Layers received different token counts')
        entry = self._cache[index]
        try:
            is_step = entry.tokens > 0 and length == 1
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
            y = scan(x, dt, aa, bm, cm, mx.D, mx.dt_bias, entry.state,
                     self.permutations[index], self.table_kind)
            if y.dtype != torch.float16 or tuple(y.shape) != (batch, length, 128, 64):
                raise RuntimeError('Dense codec returned wrong readout dtype/shape')
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

    def assert_finite_cache(self):
        for entry in self._cache:
            validate_finite_result(entry.state)
        return True

    def cache_breakdown(self):
        conv = sum(v.conv.numel()*v.conv.element_size() for v in self._cache)
        codes = sum(v.state.tensors['codes'].numel() for v in self._cache)
        scales = sum(v.state.tensors['scales'].numel()*2 for v in self._cache)
        state = sum(v.state.nbytes for v in self._cache)
        if state != codes + scales:
            raise RuntimeError('Dense state storage differs from payload count')
        tables = self.permutations.numel()*self.permutations.element_size()
        return {'mode': self.mode, 'table_kind': self.table_kind, 'batch_size': self._batch_size,
                'allocated_layers': len(self._cache), 'conv_fp16_bytes': conv,
                'ssm_payload_bytes': codes, 'ssm_scale_bytes': scales,
                'ssm_total_bytes': state, 'table_bytes': tables,
                'total_bytes': conv+state+tables, 'calibration_workspace_bytes': 0,
                'tokens_per_layer': [v.tokens for v in self._cache],
                'scope': 'Persistent request cache plus one compact table; weights, activations, '
                         'registers, allocator reserve and training workspace excluded'}


DenseStateQuant = Dense3StateQuant
