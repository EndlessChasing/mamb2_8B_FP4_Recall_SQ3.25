"""Experimental post-D Resurface-inspired adapter for the frozen native runtime.

This is NOT the original pre-D Resurface insertion: native SSD already includes
D*x when norm's prehook sees y. No scan, norm, projection or cache is replaced.
Each layer applies y + sigmoid(w*u+b)*g*(V@y), without EMA or any extra state.

Usage (keep the bank alive through backward, including checkpoint replay):
    bank = ResurfaceNative(model, gate_mode='soft', expected_base_hashes=hashes)
    hidden, gates = bank.forward_hidden(ids, use_checkpoint=True)
    loss = task_loss(hidden) + gates.closure_loss() + gates.opening_loss(mask)
    loss.backward()
    bank.export_fp16(path, binding={'base_manifest_sha256': ...})
    bank.close()
    with install_fp16(model, path, expected_base_hashes=hashes) as deployed:
        logits = model(ids).logits  # Same gate mode for every evaluation task.

The external ModuleDict never becomes a child of the base model. Hooks support
native prefill and native model(..., inference_params=...) recurrent decoding;
calling mixer.step() directly bypasses the mixer-input hook and is unsupported.
Gate capture is returned by EACH checkpoint invocation, never appended during
replay. Hooks are synchronous/non-reentrant; concurrent forwards fail closed.
Hard mode uses threshold(logit > 0), with a sigmoid STE only during training.
V=0 plus a closed hard gate is a dead point for answer CE alone: explicit opening
loss or an independently declared soft warmup is required. This module chooses
neither a training recipe nor a quality gate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import weakref

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FORMAT = 'MAMBA2_POST_D_RESURFACE_FP16_V1'
FIELDS = ('V_read', 'g_read', 'router_w', 'router_b')
VARIANT = 'post-D native norm-prehook; memoryless cross-head mixing'
_OWNERS = weakref.WeakKeyDictionary()


def tensor_hash(value):
    value = value.detach()
    digest = hashlib.sha256()
    if value.ndim == 0:
        digest.update(value.cpu().contiguous().numpy().tobytes())
    else:
        for first in range(0, value.shape[0], 128):
            digest.update(value[first:first+128].cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GateCapture:
    """FP32 logits [layers,batch,tokens]; all terms are differentiable."""
    logits: torch.Tensor

    def soft_probabilities(self):
        return torch.sigmoid(self.logits)

    def closure_loss(self, budget=0.006, budget_coefficient=10.0):
        if not 0 <= budget <= 1 or budget_coefficient < 0:
            raise ValueError('Invalid closure budget/coefficient')
        # Mean BCE and per-layer mean-openness penalty, then mean over layers.
        close = F.softplus(self.logits).mean(dim=(1, 2))
        openness = self.soft_probabilities().mean(dim=(1, 2))
        return (close + budget_coefficient * F.relu(openness-budget)).mean()

    def opening_loss(self, mask):
        """Mean BCE(logit,1) only at explicit TRAIN query-token positions."""
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(self.logits.shape[1:]):
            raise ValueError('Opening mask must be bool[batch,tokens]')
        if mask.device != self.logits.device or not bool(mask.any()):
            raise ValueError('Opening mask must share device and select tokens')
        return F.softplus(-self.logits[:, mask]).mean()


class _Adapter(nn.Module):
    def __init__(self, width, heads, *, device, trainable):
        super().__init__()
        dtype = torch.float32 if trainable else torch.float16
        for name, value in {
            'V_read': torch.zeros(heads, heads, device=device, dtype=dtype),
            'g_read': torch.ones(heads, device=device, dtype=dtype),
            'router_w': torch.zeros(width, device=device, dtype=dtype),
            'router_b': torch.tensor(-4.0, device=device, dtype=dtype),
        }.items():
            self.register_parameter(name, nn.Parameter(value, requires_grad=trainable))

    def forward(self, y, u, heads, head_dim, mode, trainable):
        # Cast INSIDE the hook, hence inside every checkpoint replay.
        V, g, w, b = (getattr(self, name).to(torch.float16) for name in FIELDS)
        logits = (torch.einsum('...d,d->...', u, w) + b).float()
        soft = torch.sigmoid(logits)
        if mode == 'soft':
            gate = soft
        else:
            hard = (logits > 0).to(soft.dtype)
            gate = hard + (soft-soft.detach()) if trainable and torch.is_grad_enabled() else hard
        yh = y.reshape(*y.shape[:-1], heads, head_dim)
        mix = torch.einsum('ij,...jp->...ip', V, yh)
        correction = gate.to(y.dtype)[..., None, None] * g[..., None] * mix
        return (yh+correction).reshape_as(y), logits


class ResurfaceNative:
    """Separate adapter parameters; original model keeps its exact parameter set."""
    def __init__(self, model, gate_mode='soft', *, expected_base_hashes=None,
                 trainable=True, production=True):
        if gate_mode not in ('soft', 'hard'):
            raise ValueError('Gate mode must be explicitly soft or hard')
        if model in _OWNERS and _OWNERS[model]() is not None:
            raise RuntimeError('A Resurface adapter is already attached to this model')
        self.model, self._gate_mode = model, gate_mode
        self.trainable, self.production = bool(trainable), bool(production)
        self.adapters = nn.ModuleDict()
        self._frames, self._requests, self._handles = {}, {}, []
        self.closed = False
        self._base = dict(model.named_parameters())
        if any(p.dtype != torch.float16 or p.requires_grad or p.grad is not None for p in self._base.values()):
            raise ValueError('Load an already-frozen FP16 base with no gradients')
        if production and (len(self._base) != 507 or sum(p.numel() for p in self._base.values()) != 8236999680):
            raise ValueError('Production base must cover507 tensors/8,236,999,680 parameters')
        self._identities = {name: (id(p), p.data_ptr(), p._version) for name, p in self._base.items()}
        self._base_hashes = dict(expected_base_hashes) if expected_base_hashes is not None else None
        if self._base_hashes is not None and set(self._base_hashes) != set(self._base):
            raise ValueError('Expected base ledger must cover every original parameter')
        backbone = model.backbone
        if backbone.fused_add_norm or backbone.residual_in_fp32:
            raise ValueError('Only frozen unfused FP16 residual orchestration is supported')
        if production and len(backbone.layers) != 56:
            raise ValueError('Production layer count must be56')
        self.geometry = []
        for index, layer in enumerate(backbone.layers):
            mx = layer.mixer
            if layer.fused_add_norm or layer.residual_in_fp32 or layer.mlp is not None:
                raise ValueError('Unsupported native block')
            if mx.use_mem_eff_path or not mx.rmsnorm:
                raise ValueError('Native explicit gated norm must execute on every call')
            width, heads, head_dim = mx.d_model, mx.nheads, mx.headdim
            if mx.d_ssm != heads*head_dim:
                raise ValueError('Unsupported head geometry')
            if production and (width, heads, head_dim, mx.ngroups) != (4096, 128, 64, 8):
                raise ValueError('Production Mamba2 geometry differs')
            self.geometry.append({'width': width, 'heads': heads, 'head_dim': head_dim})
            self.adapters[f'layer{index}'] = _Adapter(width, heads, device=mx.in_proj.weight.device,
                                                     trainable=self.trainable)
        self.masters = {f'{key}.{name}': p for key, module in self.adapters.items()
                        for name, p in module.named_parameters()}
        _OWNERS[model] = weakref.ref(self)
        try:
            for index, layer in enumerate(backbone.layers):
                self._handles.append(layer.mixer.register_forward_pre_hook(self._input_hook(index), with_kwargs=True))
                self._handles.append(layer.mixer.norm.register_forward_pre_hook(self._norm_hook(index), with_kwargs=True))
                self._handles.append(layer.mixer.register_forward_hook(self._end_hook(index), with_kwargs=True, always_call=True))
        except Exception:
            self.close()
            raise

    def parameters(self):
        return list(self.masters.values())

    @property
    def gate_mode(self):
        return self._gate_mode

    def _require_open(self):
        if self.closed:
            raise RuntimeError('Adapter was closed before this invocation/replay')

    def _input_hook(self, index):
        def hook(module, args, kwargs):
            self._require_open()
            if index in self._frames:
                raise RuntimeError('Concurrent/reentrant mixer invocation is unsupported')
            u = args[0] if args else kwargs.get('u')
            if not isinstance(u, torch.Tensor) or u.ndim not in (2, 3):
                raise ValueError('Mixer input must be rank2/3 tensor')
            if u.dtype != torch.float16 or u.shape[-1] != self.geometry[index]['width']:
                raise ValueError('Mixer input geometry/dtype differs')
            frame = self._requests.get(index, {})
            frame.update(u=u, logits=None, consumed=False)
            self._frames[index] = frame
        return hook

    def _norm_hook(self, index):
        def hook(module, args, kwargs):
            frame = self._frames.get(index)
            if frame is None or frame['consumed']:
                raise RuntimeError('Norm hook has no unique corresponding mixer input')
            y = args[0] if args else kwargs.get('x')
            geo = self.geometry[index]
            if not isinstance(y, torch.Tensor) or y.ndim not in (2, 3) or y.dtype != torch.float16:
                raise ValueError('Native norm input must be rank2/3 FP16')
            if y.shape[-1] != geo['heads']*geo['head_dim']:
                raise ValueError('Native norm/head shape differs')
            u = frame['u']
            if u.numel()//geo['width'] != y.numel()//y.shape[-1]:
                raise ValueError('Mixer input and readout token counts differ')
            u = u.reshape(*y.shape[:-1], geo['width'])
            adapted, logits = self.adapters[f'layer{index}'](y, u, geo['heads'], geo['head_dim'],
                                                            self.gate_mode, self.trainable)
            frame.update(u=None, consumed=True, logits=logits)
            if args:
                return (adapted, *args[1:]), kwargs
            return args, {**kwargs, 'x': adapted}
        return hook

    def _end_hook(self, index):
        def hook(module, args, kwargs, output):
            frame = self._frames.pop(index, None)
            if frame is not None:
                frame['u'] = None
                if output is not None and not frame['consumed']:
                    raise RuntimeError('Native mixer bypassed adapter injection')
        return hook

    @contextmanager
    def _capture(self, index):
        self._require_open()
        if index in self._requests or index in self._frames:
            raise RuntimeError('Overlapping gate capture')
        frame = {}
        self._requests[index] = frame
        try:
            yield frame
        finally:
            self._requests.pop(index, None)
            active = self._frames.pop(index, None)
            if active is not None:
                active['u'] = None
            frame.pop('u', None)

    def forward_hidden(self, input_ids, use_checkpoint=True):
        """Native blocks/final norm; only adapter leaves receive gradients."""
        self._require_open()
        hidden = self.model.backbone.embedding(input_ids)
        residual, all_logits = None, []
        for index, layer in enumerate(self.model.backbone.layers):
            def run(h, r, index=index, layer=layer):
                with self._capture(index) as frame:
                    h, r = layer(h, r, inference_params=None)
                    if frame.get('logits') is None:
                        raise RuntimeError('Gate capture missing from this block invocation')
                    return h, r, frame['logits']
            if use_checkpoint and torch.is_grad_enabled():
                hidden, residual, logits = checkpoint(run, hidden, residual, use_reentrant=False,
                                                       preserve_rng_state=True)
            else:
                hidden, residual, logits = run(hidden, residual)
            all_logits.append(logits)  # Once per OUTER layer, never inside replay.
        residual = hidden+residual if residual is not None else hidden
        hidden = self.model.backbone.norm_f(residual.to(self.model.backbone.norm_f.weight.dtype))
        return hidden, GateCapture(torch.stack(all_logits))

    def state_dict(self):
        return {name: p.detach().cpu().clone() for name, p in self.masters.items()}

    def load_state_dict(self, values):
        dtype = torch.float32 if self.trainable else torch.float16
        if set(values) != set(self.masters):
            raise ValueError('Adapter keys differ')
        for name, value in values.items():
            if value.shape != self.masters[name].shape or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError(f'Invalid adapter tensor: {name}')
        with torch.no_grad():
            for name, value in values.items():
                self.masters[name].copy_(value)

    def assert_base_frozen(self, check_values=False):
        now = dict(self.model.named_parameters())
        if set(now) != set(self._base):
            raise RuntimeError('Base parameter inventory changed')
        for name, p in now.items():
            if (id(p), p.data_ptr(), p._version) != self._identities[name] or p.requires_grad or p.grad is not None:
                raise RuntimeError(f'Base identity/version/gradient changed: {name}')
        if check_values:
            if self._base_hashes is None:
                raise ValueError('Content audit requires an independently supplied base hash ledger')
            for name, p in now.items():
                if tensor_hash(p) != self._base_hashes[name]:
                    raise RuntimeError(f'Base content changed: {name}')
        return {'tensors': len(now), 'parameters': sum(p.numel() for p in now.values()),
                'identity_version_gradients_unchanged': True, 'actual_content_checked': check_values}

    def export_fp16(self, path, binding=None):
        self.assert_base_frozen()
        values = {name: p.detach().cpu().half() for name, p in self.masters.items()}
        if not all(torch.isfinite(v).all() for v in values.values()):
            raise FloatingPointError('Nonfinite FP16 export')
        payload = {'format': FORMAT, 'variant': VARIANT, 'gate_mode': self.gate_mode,
                   'geometry': self.geometry, 'binding': dict(binding or {}), 'tensors': values}
        path = Path(path)
        with path.open('xb') as stream:
            torch.save(payload, stream)
        restored = read_fp16(path, expected_binding=payload['binding'])
        hashes = {name: tensor_hash(value) for name, value in values.items()}
        if hashes != {name: tensor_hash(value) for name, value in restored['tensors'].items()}:
            raise RuntimeError('Actual FP16 file roundtrip differs')
        return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': file_hash(path),
                'parameters': sum(v.numel() for v in values.values()),
                'payload_bytes': sum(v.numel()*v.element_size() for v in values.values()),
                'tensor_sha256': hashes, 'gate_mode': self.gate_mode, 'roundtrip_bitwise_equal': True}

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._frames.clear()
        self._requests.clear()
        if self.model in _OWNERS and _OWNERS[self.model]() is self:
            del _OWNERS[self.model]
        self.closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()


def read_fp16(path, expected_binding=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (set(payload) != {'format', 'variant', 'gate_mode', 'geometry', 'binding', 'tensors'}
            or payload['format'] != FORMAT or payload['variant'] != VARIANT
            or payload['gate_mode'] not in ('soft', 'hard') or not isinstance(payload['binding'], dict)):
        raise ValueError('Invalid Resurface export header')
    if expected_binding is not None and payload['binding'] != expected_binding:
        raise ValueError('Resurface binding differs')
    expected = {}
    for index, geo in enumerate(payload['geometry']):
        if set(geo) != {'width', 'heads', 'head_dim'} or any(type(v) is not int or v <= 0 for v in geo.values()):
            raise ValueError('Invalid export geometry')
        expected.update({f'layer{index}.{name}': shape for name, shape in {
            'V_read': (geo['heads'], geo['heads']), 'g_read': (geo['heads'],),
            'router_w': (geo['width'],), 'router_b': (),
        }.items()})
    if not expected or set(payload['tensors']) != set(expected):
        raise ValueError('Export tensor inventory differs')
    for name, shape in expected.items():
        v = payload['tensors'][name]
        if not isinstance(v, torch.Tensor) or v.dtype != torch.float16 or tuple(v.shape) != shape or not torch.isfinite(v).all():
            raise ValueError(f'Invalid exported tensor: {name}')
    return payload


def install_fp16(model, path, *, expected_binding=None, expected_base_hashes=None, production=True):
    """Fresh eval-only adapter loaded from actual serialized half tensors."""
    payload = read_fp16(path, expected_binding)
    bank = ResurfaceNative(model, payload['gate_mode'], expected_base_hashes=expected_base_hashes,
                           trainable=False, production=production)
    try:
        if bank.geometry != payload['geometry']:
            raise ValueError('Export does not fit actual native model')
        bank.load_state_dict(payload['tensors'])
    except Exception:
        bank.close()
        raise
    return bank


def self_test():
    """Bounded CPU hook tests; not an 8B native-kernel or quality validation."""
    torch.set_num_threads(8)
    torch.manual_seed(9231)

    class Norm(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width, dtype=torch.float16))
        def forward(self, x, z=None):
            if z is not None:
                x = x*F.silu(z)
            return (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-5)).half()*self.weight

    class Mixer(nn.Module):
        d_model, d_ssm, nheads, headdim, ngroups = 4, 4, 2, 2, 1
        use_mem_eff_path, rmsnorm = False, True
        def __init__(self, index):
            super().__init__()
            self.index, self.fail = index, False
            self.in_proj = nn.Linear(4, 8, bias=False, dtype=torch.float16)
            self.norm = Norm(4)
            self.out_proj = nn.Linear(4, 4, bias=False, dtype=torch.float16)
        def forward(self, u, inference_params=None):
            if self.fail:
                raise RuntimeError('intentional toy mixer failure')
            if inference_params is not None and u.shape[1] == 1:
                return self.step(u, inference_params)
            x, z = self.in_proj(u).chunk(2, -1)
            if inference_params is not None:
                inference_params[self.index] = x[:, -1].clone()
            return self.out_proj(self.norm(x, z))
        def step(self, u, cache):
            x, z = self.in_proj(u[:, 0]).chunk(2, -1)
            x = x+cache.get(self.index, torch.zeros_like(x))*0.125
            cache[self.index] = x.clone()
            return self.out_proj(self.norm(x, z)).unsqueeze(1)

    class Block(nn.Module):
        fused_add_norm, residual_in_fp32, mlp = False, False, None
        def __init__(self, index):
            super().__init__()
            self.norm, self.mixer = Norm(4), Mixer(index)
        def forward(self, hidden, residual=None, inference_params=None):
            residual = hidden+residual if residual is not None else hidden
            return self.mixer(self.norm(residual), inference_params=inference_params), residual

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.fused_add_norm = self.backbone.residual_in_fp32 = False
            self.backbone.embedding = nn.Embedding(17, 4, dtype=torch.float16)
            self.backbone.layers = nn.ModuleList([Block(i) for i in range(2)])
            self.backbone.norm_f = Norm(4)
            self.lm_head = nn.Linear(4, 17, bias=False, dtype=torch.float16)
        def forward(self, ids, inference_params=None):
            h, r = self.backbone.embedding(ids), None
            for layer in self.backbone.layers:
                h, r = layer(h, r, inference_params=inference_params)
            return self.lm_head(self.backbone.norm_f(h+r))

    model = Model().eval().requires_grad_(False)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    hashes = {name: tensor_hash(p) for name, p in model.named_parameters()}
    tests = []
    def passed(name): tests.append(name)
    def rejects(call):
        try: call()
        except (ValueError, RuntimeError, FileExistsError): return
        raise AssertionError('Expected fail-closed rejection')
    def zero_grads(bank):
        for p in bank.parameters(): p.grad = None
    def cached():
        cache = {}
        prefill = model(ids[:, :4], inference_params=cache)
        step = model(ids[:, 4:5], inference_params=cache)
        return tensor_hash(prefill), tensor_hash(step), {k: tensor_hash(v) for k, v in cache.items()}

    with torch.no_grad():
        baseline, base_cache = tensor_hash(model(ids)), cached()
    bank = ResurfaceNative(model, expected_base_hashes=hashes, production=False)
    with torch.no_grad():
        assert tensor_hash(model(ids)) == baseline and cached() == base_cache
    bank.assert_base_frozen(check_values=True)
    passed('zero soft adapter native prefill/recurrent-call/cache parity; external inventory')

    hidden, gates = bank.forward_hidden(ids, use_checkpoint=False)
    model.lm_head(hidden).float().square().mean().backward()
    for name, p in bank.masters.items():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert bool(torch.count_nonzero(p.grad)) == name.endswith('.V_read'), name
    passed('soft zero-V staggered gradients: V nonzero; router and amplitudes zero')

    with torch.no_grad():
        for key, adapter in bank.adapters.items():
            adapter.V_read.copy_(torch.tensor([[.0713, -.0927], [.1319, .0423]]))
            adapter.router_w.copy_(torch.tensor([.0371, -.0447, .0233, .0631]))
    mask = torch.tensor([[False, False, False, False, False, True, True]])
    outputs, gradients = [], []
    for checkpointed in (False, True):
        zero_grads(bank)
        hidden, gates = bank.forward_hidden(ids, use_checkpoint=checkpointed)
        assert gates.logits.shape == (2, 1, 7)
        manual = F.binary_cross_entropy_with_logits(gates.logits[:, mask], torch.ones_like(gates.logits[:, mask]))
        torch.testing.assert_close(gates.opening_loss(mask), manual, rtol=0, atol=0)
        loss = model.lm_head(hidden).float().square().mean()+.03*gates.closure_loss()+.02*gates.opening_loss(mask)
        loss.backward()
        outputs.append((tensor_hash(hidden), tensor_hash(gates.logits)))
        gradients.append({name: p.grad.detach().clone() for name, p in bank.masters.items()})
        assert not bank._frames and not bank._requests
    assert outputs[0] == outputs[1]
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
        assert torch.isfinite(gradients[1][name]).all() and torch.count_nonzero(gradients[1][name]), name
    with torch.no_grad():
        hidden, _ = bank.forward_hidden(ids, use_checkpoint=False)
        assert tensor_hash(model.lm_head(hidden)) == tensor_hash(model(ids))
    passed('checkpoint/native outputs, returned capture and every adapter gradient exact')

    model.backbone.layers[0].mixer.fail = True
    rejects(lambda: bank.forward_hidden(ids))
    assert not bank._frames and not bank._requests
    model.backbone.layers[0].mixer.fail = False
    rejects(lambda: gates.opening_loss(torch.zeros_like(mask)))
    rejects(lambda: ResurfaceNative(model, production=False))
    passed('exception cleanup, empty opening mask and duplicate attachment rejected')

    with tempfile.TemporaryDirectory(prefix='resurface-native-cpu-') as temporary:
        path = Path(temporary)/'adapter.pt'
        with torch.no_grad(): expected = tensor_hash(model(ids))
        receipt = bank.export_fp16(path, binding={'toy_base': 'fixed'})
        expected_values = {name: p.detach().half().clone() for name, p in bank.masters.items()}
        rejects(lambda: bank.export_fp16(path))
        bank.close()
        with install_fp16(model, path, expected_binding={'toy_base': 'fixed'}, expected_base_hashes=hashes,
                          production=False) as deployed:
            assert not any(p.requires_grad for p in deployed.parameters())
            assert all(p.dtype == torch.float16 for p in deployed.parameters())
            for name, p in deployed.masters.items(): assert tensor_hash(p) == tensor_hash(expected_values[name])
            with torch.no_grad(): assert tensor_hash(model(ids)) == expected
            deployed.assert_base_frozen(check_values=True)
        assert tensor_hash(model(ids)) == baseline
        invalid = read_fp16(path)
        invalid['tensors']['layer0.router_b'] = torch.tensor(float('nan'), dtype=torch.float16)
        bad = Path(temporary)/'bad.pt'
        torch.save(invalid, bad)
        rejects(lambda: read_fp16(bad))
        passed('real FP16 file roundtrip/native parity/no overwrite/nonfinite rejection/hook removal')

    with ResurfaceNative(model, gate_mode='hard', expected_base_hashes=hashes, production=False) as hard:
        hidden, gates = hard.forward_hidden(ids)
        model.lm_head(hidden).float().square().mean().backward()
        assert all(p.grad is not None and not torch.count_nonzero(p.grad) for p in hard.parameters())
        zero_grads(hard)
        _, gates = hard.forward_hidden(ids)
        gates.opening_loss(mask).backward()
        for name, p in hard.masters.items():
            if '.router_' in name: assert p.grad is not None and torch.count_nonzero(p.grad)
        with torch.no_grad():
            for adapter in hard.adapters.values():
                adapter.V_read.fill_(.25)
            assert tensor_hash(model(ids)) == baseline
            for adapter in hard.adapters.values(): adapter.router_b.fill_(1.)
            assert tensor_hash(model(ids)) != baseline
        hard.assert_base_frozen(check_values=True)
    passed('hard closed-gate dead point, explicit opening gradients and threshold inference')
    assert tensor_hash(model(ids)) == baseline and not _OWNERS
    rejects(lambda: ResurfaceNative(model, gate_mode='task_detector', production=False))
    passed('no task-specific gate mode; all base tensors/content unchanged')
    return {'complete': True, 'tests': tests, 'passed': len(tests), 'cuda_initialized': torch.cuda.is_initialized(),
            'production_trainables': 56*(128*128+128+4096+1), 'production_fp16_payload_bytes': 2308208,
            'toy_export_bytes': receipt['bytes'], 'scope': 'CPU toy hooks only; native8B GPU parity/quality unmeasured'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2))
