"""Full-vocabulary staged head gradients plus one joint backbone/gate backward.

MK selects every answer-suffix target with an explicit bool mask; prose selects
all targets. Only loss-position selection uses the mask: it is never a router
input. Chunk sums divide by the complete selected target count. Head GEMMs are
FP16, CE/KL are FP32, temperature is one, teacher is detached. The scaler applies
once to the hidden derivative and once to the separate gate-auxiliary branch.
"""
from __future__ import annotations
import argparse
import json
import math
import torch
import torch.nn.functional as F


def staged_loss_backward(student_hidden, targets, student_head, *, gates=None,
                         scaler=None, teacher_hidden=None, teacher_head=None,
                         answer_mask=None, ce_weight=1., kl_weight=0.,
                         closure_weight=0., opening_mask=None, opening_weight=0.,
                         chunk_tokens=64):
    weights = (ce_weight, kl_weight, closure_weight, opening_weight)
    if any(not math.isfinite(x) or x < 0 for x in weights) or sum(weights) <= 0:
        raise ValueError('Invalid objective coefficients')
    if (student_hidden.ndim != 3 or student_hidden.dtype != torch.float16
            or targets.dtype != torch.int64 or targets.shape != student_hidden.shape[:2]
            or targets.device != student_hidden.device or not student_hidden.requires_grad
            or type(chunk_tokens) is not int or chunk_tokens < 1):
        raise ValueError('Invalid native FP16 hidden/target/chunk geometry')
    if answer_mask is None:
        answer_mask = torch.ones_like(targets, dtype=torch.bool)
    if (answer_mask.dtype != torch.bool or answer_mask.shape != targets.shape
            or answer_mask.device != targets.device or not bool(answer_mask.any())):
        raise ValueError('Answer selection must be nonempty bool[batch,tokens]')
    count = int(answer_mask.sum())
    if not torch.isfinite(student_hidden).all():
        raise FloatingPointError('Nonfinite student hidden')
    if any(p.requires_grad for p in student_head.parameters()):
        raise ValueError('Student vocabulary head must be frozen')
    if kl_weight:
        if (teacher_hidden is None or teacher_head is None or teacher_hidden.shape != student_hidden.shape
                or teacher_hidden.dtype != torch.float16 or teacher_hidden.device != student_hidden.device
                or not torch.isfinite(teacher_hidden).all()
                or any(p.requires_grad for p in teacher_head.parameters())):
            raise ValueError('Invalid frozen teacher')
    if (closure_weight or opening_weight) and gates is None:
        raise ValueError('Gate objective requires same-invocation capture')
    if gates is not None and (gates.logits.shape[1:] != targets.shape or gates.logits.device != targets.device):
        raise ValueError('Gate capture does not correspond to this hidden sequence')
    if gates is not None and not torch.isfinite(gates.logits).all():
        raise FloatingPointError('Nonfinite router logits, including sigmoid-saturated values')
    gradient = torch.zeros_like(student_hidden)
    ce_parts, kl_parts, sizes = [], [], []
    with torch.autocast(device_type=student_hidden.device.type, enabled=False):
        for first in range(0, student_hidden.shape[1], chunk_tokens):
            last = min(first+chunk_tokens, student_hidden.shape[1])
            mask = answer_mask[:, first:last]
            selected = int(mask.sum())
            if not selected:
                continue
            chunk = student_hidden[:, first:last].detach().requires_grad_(True)
            logits16 = student_head(chunk[mask])
            if logits16.dtype != torch.float16 or not torch.isfinite(logits16).all():
                raise FloatingPointError('Invalid native student head output')
            logits = logits16.float()
            ce = F.cross_entropy(logits, targets[:, first:last][mask], reduction='sum')
            kl = logits.new_zeros(())
            if kl_weight:
                with torch.no_grad():
                    teacher_logits16 = teacher_head(teacher_hidden[:, first:last][mask])
                    if (teacher_logits16.dtype != torch.float16 or teacher_logits16.shape != logits16.shape
                            or not torch.isfinite(teacher_logits16).all()):
                        raise FloatingPointError('Invalid native teacher head output')
                    teacher_logp = F.log_softmax(teacher_logits16.float(), dim=-1)
                kl = F.kl_div(F.log_softmax(logits, dim=-1), teacher_logp, reduction='sum', log_target=True)
            loss = (ce_weight*ce+kl_weight*kl)/count
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite chunk objective')
            scaled = scaler.scale(loss) if scaler is not None else loss
            grad, = torch.autograd.grad(scaled, chunk)
            gradient[:, first:last].copy_(grad)
            ce_parts.append(float(ce.detach())); kl_parts.append(float(kl.detach())); sizes.append(selected)
        auxiliary, close, opening = None, 0., 0.
        if closure_weight:
            term = gates.closure_loss()
            close = float(term.detach())
            auxiliary = closure_weight*term
        if opening_weight:
            term = gates.opening_loss(opening_mask)
            opening = float(term.detach())
            term = opening_weight*term
            auxiliary = term if auxiliary is None else auxiliary+term
        if auxiliary is not None:
            if not torch.isfinite(auxiliary):
                raise FloatingPointError('Nonfinite gate objective')
            scaled_auxiliary = scaler.scale(auxiliary) if scaler is not None else auxiliary
            torch.autograd.backward((student_hidden, scaled_auxiliary), (gradient, None))
        else:
            student_hidden.backward(gradient)
    ce_mean, kl_mean = math.fsum(ce_parts)/count, math.fsum(kl_parts)/count
    return {'targets': count, 'forward_positions': targets.numel(), 'chunks': len(sizes),
            'chunk_selected_targets': sizes, 'ce': ce_mean, 'teacher_to_student_kl': kl_mean,
            'closure': close, 'opening': opening, 'weighted_closure': closure_weight*close,
            'weighted_opening': opening_weight*opening,
            'loss': ce_weight*ce_mean+kl_weight*kl_mean+closure_weight*close+opening_weight*opening,
            'ce_coefficient': ce_weight, 'kl_coefficient': kl_weight, 'temperature': 1.,
            'scaled_hidden_gradient_finite': bool(torch.isfinite(gradient).all()),
            'scaled_hidden_gradient_nonzero': int(torch.count_nonzero(gradient)),
            'scaled_hidden_gradient_elements': gradient.numel(),
            'backbone_backward_calls': 1, 'gate_and_hidden_backward_joint': auxiliary is not None}


def self_test():
    from .resurface_native import GateCapture
    torch.manual_seed(3289)
    torch.set_num_threads(8)
    head = torch.nn.Linear(5, 19, bias=False, dtype=torch.float16).requires_grad_(False)
    targets = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    seed = torch.randn(1, 7, 5)*.25
    teacher = (seed+.1).half()
    mask = torch.tensor([[False, False, True, True, False, True, True]])
    opening_mask = torch.tensor([[False, False, False, False, False, True, True]])
    class Scale:
        def scale(self, loss): return 8*loss
    tests = []
    for label, selected, klw, closew, openw in [('masked MK', mask, 0., 0., .02),
                                               ('prose', torch.ones_like(mask), .5, 3., 0.)]:
        gradients, losses = [], []
        cew = .5 if klw else 1.
        for staged in (False, True):
            master = seed.clone().requires_grad_()
            hidden = master.half()
            gates = GateCapture(master.mean(-1).unsqueeze(0))
            if staged:
                row = staged_loss_backward(hidden, targets, head, gates=gates, scaler=Scale(),
                    teacher_hidden=teacher, teacher_head=head, answer_mask=selected, ce_weight=cew,
                    kl_weight=klw, closure_weight=closew, opening_mask=opening_mask,
                    opening_weight=openw, chunk_tokens=3)
                losses.append(row['loss'])
                assert row['targets'] == int(selected.sum()) and row['chunk_selected_targets'][-1] == 1
                assert row['backbone_backward_calls'] == 1 and row['scaled_hidden_gradient_finite']
            else:
                logits = head(hidden[selected]).float()
                ce = F.cross_entropy(logits, targets[selected], reduction='sum')/selected.sum()
                with torch.no_grad(): teacher_logp = F.log_softmax(head(teacher[selected]).float(), -1)
                kl = F.kl_div(F.log_softmax(logits, -1), teacher_logp, reduction='sum', log_target=True)/selected.sum()
                loss = cew*ce+klw*kl+closew*gates.closure_loss()+openw*gates.opening_loss(opening_mask)
                (8*loss).backward(); losses.append(float(loss.detach()))
            gradients.append(master.grad)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=.002, atol=.0001)
        assert abs(losses[0]-losses[1]) < 2e-6
        tests.append(label+': direct gradient, short tail, gate branch and scale once')
    # Non-answer labels cannot affect masked-MK objective or gradient.
    rows = []
    for changed in (False, True):
        master = seed.clone().requires_grad_()
        modified = targets.clone()
        if changed: modified[~mask] = (modified[~mask]+4)%19
        receipt = staged_loss_backward(master.half(), modified, head, answer_mask=mask, chunk_tokens=3)
        rows.append((receipt['loss'], master.grad))
    assert rows[0][0] == rows[1][0] and torch.equal(rows[0][1], rows[1][1])
    tests.append('Prompt-label changes cannot affect answer-only CE')
    invalid = GateCapture(torch.full((1, 1, 7), float('inf')))
    try:
        staged_loss_backward(seed.clone().requires_grad_().half(), targets, head, gates=invalid, answer_mask=mask)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Nonfinite gate logits accepted by MK with zero closure weight')
    tests.append('Reject nonfinite router logits even when MK closure coefficient is zero')
    return {'complete': True, 'tests': tests, 'passed': len(tests),
            'cuda_initialized': torch.cuda.is_initialized(), 'scope': 'CPU small vocabulary loss algebra only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2))
