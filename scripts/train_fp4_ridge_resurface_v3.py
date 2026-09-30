#!/usr/bin/env python3
"""Fresh Resurface training on frozen FP4 G16 group-ridge state; explicit surrogate gradients."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall import resurface_data as data, resurface_native as native, runtime
from mamba2_recall import resurface_loss as objective
from mamba2_recall.fp4 import load_fp4_model
from mamba2_recall.state_quant import StateQuant
from state_ridge_training_v3 import StateRidgeTrainingV3
from fp4_zero_predictor_codec_v1 import PredictorState
import ridge_resurface_binding_v3 as shared
from state_ppl_codec_v10 import StatePPLQuantV10
from train_quant_first import (STEPS, PROSE_MANIFEST_SHA, PROSE_TOKENS_SHA,
                               optimizer_for, pair_for, schedule, lr_factor)
from train_resurface_more import check_optimizer, equal_tree, expected_scaler_after
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
import fp4_state_binding_v1 as fp4
from run_fp4_state_quality_v1 import assert_weight_content

FORMAT = 'FP4_RIDGE_RESURFACE_TRAIN_V3'
PROTOCOL = fp4.sha(ROOT / 'docs/FP4_G16_RIDGE_RESURFACE_TRAIN_V3_PROTOCOL.md')
BASELINE = ROOT / 'artifacts/fp4_weight_v1/fp4_g16_v1'
BASELINE_AUDIT = ROOT / 'artifacts/fp4_weight_v1/fp4_g16_audit_v1.json'
TRAIN_SHA = '451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0'
OBJECTIVE = dict(numeric_ce=.25, prose_ce=1., prose_kl=.5, prose_closure=.1,
                 closure_budget=.006, closure_budget_coefficient=10., temperature=1.)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def hashes():
    return shared.hashes(extra=(
        'scripts/train_fp4_ridge_resurface_v3.py',
        'scripts/audit_train_fp4_ridge_resurface_v3.py',
        'scripts/state_ridge_training_v3.py','scripts/check_state_ridge_training_v3.py',
        'scripts/state_ppl_training_scan_v11.py','scripts/state_ppl_training_v11.py',
        'scripts/train_quant_first.py','scripts/train_resurface_more.py',
        'scripts/evaluate_resurface_more.py','mamba2_recall/state_training.py',
        'mamba2_recall/resurface_loss.py','mamba2_recall/resurface_data.py',
        'docs/FP4_G16_RIDGE_RESURFACE_TRAIN_V3_PROTOCOL.md'))


def inputs(args):
    parent = shared.load_parent(args)
    table, _, _, _, _, _, expected, provenance = parent
    fixture = fp4.read_json(args.codec_check)
    need(fixture['format']=='RIDGE_STATE_TRAINING_CHECK_V3' and fixture['passed'] is True
         and len(fixture['rows'])==42 and fixture['v2_strided_gradient_mismatch_seen'] is True and all(r['forward_exact'] is True and
         r['legacy_surrogate_allclose'] is True and r['gradient_finite'] is True
         for r in fixture['rows']), 'Exact forward/surrogate gradient fixture required')
    for name,digest in fixture['source_sha256'].items():
        need(fp4.sha(ROOT/name)==digest,'Fixture source changed')
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256 == fp4.TOKENIZER_SHA, 'Tokenizer differs')
    manifest, _, examples = data.load_training(args.data_root, TRAIN_SHA, tokenizer)
    need(fp4.sha(args.data_root / 'train/manifest.json') == TRAIN_SHA
         and len(examples) == STEPS, 'Pinned numeric TRAIN differs')
    need(fp4.sha(args.prose_manifest) == PROSE_MANIFEST_SHA
         and fp4.sha(args.prose_tokens) == PROSE_TOKENS_SHA, 'Pinned prose TRAIN differs')
    prose_manifest = fp4.read_json(args.prose_manifest)
    windows = torch.load(args.prose_tokens, map_location='cpu', weights_only=True)
    need(tuple(windows.shape) == (448, 2048) and windows.dtype == torch.int64
         and sorted(prose_manifest['schedule']) == list(range(448))
         and runtime.token_digest(windows.flatten().numpy()) ==
         prose_manifest['training_tokens_sha256_int64le'], 'Pinned prose population differs')
    binding = dict(format=FORMAT, experiment_protocol_sha256=PROTOCOL,
                   fp4_protocol_sha256=fp4.PROTOCOL_SHA, source_sha256=fp4.SOURCE_SHA,
                   ridge_parent=provenance, codec_check_sha256=fp4.sha(args.codec_check),
                   numeric_train_manifest_sha256=TRAIN_SHA,
                   prose_manifest_sha256=PROSE_MANIFEST_SHA,
                   prose_tokens_sha256=PROSE_TOKENS_SHA,
                   tokenizer_sha256=tokenizer.sha256,
                   weight_format='fp4_g16_e4m3', state_layout='mixed_ridge_sq325',
                   forward='exact frozen deployed ridge state',
                   backward='legacy32/32/64 live-mask STE; ignores ridge latent/predictor derivatives',
                   teacher='unadapted FP4 G16, S16 state',
                   initial_adapter='fresh V=0,g=1,w=0,b=-4',
                   planned_successful_updates=STEPS, objective=OBJECTIVE,
                   recipe_selected_before_training=True, recipe_selection_used_validation=False)
    return tokenizer, examples, windows, prose_manifest['schedule'], table, expected, binding, parent


def put(path, obj):
    fp4.write_json(path, obj)


def check_model(model, expected):
    receipt = model._fp4_receipt
    need(receipt['format_name'] == 'fp4_g16_e4m3'
         and receipt['physical_packed_payload_bytes'] == fp4.PAYLOAD_BYTES['fp4_g16_e4m3']
         and receipt['source_checkpoint_sha256'] == fp4.SOURCE_SHA,
         'FP4 loader provenance differs')
    return assert_weight_content(model, expected)


def load_base(args, out_dir, role, expected):
    model = load_fp4_model(args.source_dir, 'fp4_g16_e4m3', device='cuda', chunk_rows=512,
        evidence_path=out_dir / (role + '_conversion_evidence.pt'))
    proof = check_model(model, expected)
    put(out_dir / (role + '_conversion_receipt.json'), model._fp4_receipt)
    return model, proof


def attempt(bank, teacher, teacher_execution, pair, optimizer, scaler, j):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr']*lr_factor(j)
    hidden,gates = bank.forward_hidden(pair['mk_ids'], use_checkpoint=True)
    mk = objective.staged_loss_backward(hidden,pair['mk_targets'],bank.model.lm_head,
        gates=gates,scaler=scaler,answer_mask=pair['answer_mask'],ce_weight=OBJECTIVE['numeric_ce'],
        kl_weight=0.,closure_weight=0.,chunk_tokens=64)
    del hidden,gates
    with torch.no_grad():
        teacher_hidden = teacher_execution.backbone(pair['prose_ids'], reset=True)
    hidden,gates = bank.forward_hidden(pair['prose_ids'],use_checkpoint=True)
    prose = objective.staged_loss_backward(hidden,pair['prose_targets'],bank.model.lm_head,
        gates=gates,scaler=scaler,teacher_hidden=teacher_hidden,
        teacher_head=teacher.lm_head,ce_weight=OBJECTIVE['prose_ce'],kl_weight=OBJECTIVE['prose_kl'],
        closure_weight=OBJECTIVE['prose_closure'],chunk_tokens=64)
    del hidden,gates,teacher_hidden
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in bank.parameters()]
    if any(g is None for g in gradients):
        raise RuntimeError('Missing adapter gradient')
    overflow = (not all(bool(torch.isfinite(g).all()) for g in gradients)
                or not mk['scaled_hidden_gradient_finite']
                or not prose['scaled_hidden_gradient_finite'])
    if overflow:
        scaler.update(new_scale=scaler.get_scale()*.5)
        magnitude = None
    else:
        magnitude = float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.,error_if_nonfinite=True))
        scaler.step(optimizer)
        scaler.update()
        if any(not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(p.half()).all())
               for p in bank.parameters()):
            raise FloatingPointError('Nonfinite updated adapter master or FP16 cast')
    bank.assert_base_frozen()
    return {'mk_ce':mk['ce'], 'prose_ce':prose['ce'],
            'prose_kl':prose['teacher_to_student_kl'],
            'prose_closure':prose['closure'], 'overflow':overflow,
            'gradient_norm_before_clip':magnitude, 'loss_scale':scaler.get_scale(),
            'objective':OBJECTIVE,
            'weighted_numeric_ce':OBJECTIVE['numeric_ce']*mk['ce'],
            'weighted_prose_ce':OBJECTIVE['prose_ce']*prose['ce'],
            'weighted_prose_kl':OBJECTIVE['prose_kl']*prose['teacher_to_student_kl'],
            'weighted_prose_closure':OBJECTIVE['prose_closure']*prose['closure']}



def save_checkpoint(path, bank, optimizer, scaler, binding, successful, attempts):
    need(not path.exists(), 'Checkpoint path exists')
    temporary = path.with_suffix('.pt.tmp')
    need(not temporary.exists(), 'Checkpoint temp path exists')
    with temporary.open('wb') as stream:
        torch.save(dict(format='FP4_RIDGE_RESURFACE_CHECKPOINT_V3', binding=binding,
            successful_updates=successful, attempts=attempts, masters=bank.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict()), stream)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return dict(file=path.name, sha256=fp4.sha(path), bytes=path.stat().st_size,
                successful_updates=successful, attempts=attempts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source-dir', 'data-root', 'prose-manifest',
                'prose-tokens', 'out-dir', 'codec-check'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-report', type=Path)
    shared.add_parent_arguments(p)
    args = p.parse_args()
    need(args.smoke == (args.smoke_report is None),
         'Formal requires smoke report; smoke must not have one')
    need(not args.out_dir.exists(), 'Fresh output directory required')
    args.out_dir.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.manual_seed(2026092803); torch.cuda.manual_seed_all(2026092803)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    report = dict(format=FORMAT, complete=False, mode='smoke' if args.smoke else 'formal',
                  started_unix=time.time(), code_sha256=hashes(), history=[],
                  checkpoints=[], successful_updates=0, attempts=0, overflows=0)
    student_execution = teacher_execution = bank = None
    try:
        tokenizer, examples, windows, prose_order, table, expected, binding, parent = inputs(args)
        policy = pin_replay_backend()
        binding.update(code_sha256=report['code_sha256'], backend_policy=policy)
        report.update(binding=binding, backend_policy=policy)
        if args.smoke_report is not None:
            prior = fp4.read_json(args.smoke_report)
            need(prior['format'] == FORMAT and prior['complete'] is True
                 and prior['mode'] == 'smoke' and prior['successful_updates'] == 1
                 and prior['binding'] == binding and prior['code_sha256'] == report['code_sha256']
                 and prior['adapter']['discarded'] is True
                 and args.smoke_report.parent.resolve() != args.out_dir.resolve(),
                 'Independent discarded smoke differs')
            report['smoke_report_sha256'] = fp4.sha(args.smoke_report)
            report['smoke_report_path'] = str(args.smoke_report.resolve())
        put(args.out_dir / 'report.json', report)
        print('[setup] materializing two separate frozen FP4 G16 bases', flush=True)
        student, student_proof = load_base(args, args.out_dir, 'student', expected)
        teacher, teacher_proof = load_base(args, args.out_dir, 'teacher', expected)
        need(not {p.data_ptr() for p in student.parameters()} &
             {p.data_ptr() for p in teacher.parameters()}, 'Teacher and student share storage')
        report.update(initial_student_base_check=student_proof,
                      initial_teacher_base_check=teacher_proof)
        teacher_identity = {name: (id(p), p.data_ptr(), p._version)
                            for name, p in teacher.named_parameters()}
        with torch.no_grad(), PredictorState(student, *parent[:5]) as packed:
            reference = {}
            for length in (128, 512):
                ids = windows[0, :length].cuda()[None]
                hidden = packed.backbone(ids, reset=True)
                cache = packed.cache_breakdown()
                need(cache['total_bytes'] == fp4.CACHE_BYTES and bool(torch.isfinite(hidden).all()),
                     'Packed initial state/cache differs')
                reference[length] = (ids, hidden.detach().clone(), cache)
        bank = native.ResurfaceNative(student, 'soft', expected_base_hashes=expected)
        need(len(bank.masters) == 224 and sum(p.numel() for p in bank.parameters()) == 1154104,
             'Fresh adapter geometry differs')
        optimizer = optimizer_for(bank)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024., growth_factor=2.,
                                      backoff_factor=.5, growth_interval=2000)
        need(not optimizer.state and all(p.grad is None for p in bank.parameters()),
             'Fresh optimizer/gradients differ')
        for name, param in bank.masters.items():
            target = {'V_read': 0., 'g_read': 1., 'router_w': 0., 'router_b': -4.}[name.rsplit('.', 1)[-1]]
            need(bool((param == target).all()), 'Adapter is not freshly initialized')
        report['fresh_initialization'] = True
        initial_raw = {}
        student_execution = StateRidgeTrainingV3(student, *parent[:5]).install()
        teacher_execution = StateQuant(teacher, 's16').install()
        with torch.no_grad():
            for length, (ids, hidden, cache) in reference.items():
                actual, _ = bank.forward_hidden(ids, use_checkpoint=False)
                need(torch.equal(actual, hidden), f'Initial packed/training parity failed: {length}')
                initial_raw[str(length)] = dict(ids=ids.cpu(), packed_hidden=hidden.cpu(), training_hidden=actual.cpu())
        print('[setup] frozen FP4 G16 loaded; fresh adapter and packed/training parity passed', flush=True)
        ordered = schedule(); limit = 1 if args.smoke else STEPS
        successful = attempts = overflows = 0
        while successful < limit:
            index = successful
            pair = pair_for(index, ordered, examples, windows, prose_order)
            before = copy.deepcopy(scaler.state_dict())
            result = attempt(bank, teacher, teacher_execution, pair, optimizer, scaler, index)
            equal_tree(scaler.state_dict(), expected_scaler_after(before, result['overflow']),
                       'GradScaler transition')
            student_execution.assert_frozen()
            need(all((id(p), p.data_ptr(), p._version) == teacher_identity[name]
                     and p.grad is None for name, p in teacher.named_parameters()),
                 'Teacher base changed')
            attempts += 1; overflows += int(result['overflow']); successful += int(not result['overflow'])
            need(overflows <= 8, 'Overflow retry budget exceeded')
            row = dict(attempt=attempts, successful_updates=successful,
                       update_index=index, schedule_entry=pair['schedule_entry'],
                       case_id=pair['id'], prose_window=pair['prose_window'],
                       prose_start=pair['prose_start'], **result)
            report['history'].append(row)
            report.update(successful_updates=successful, attempts=attempts, overflows=overflows)
            if not args.smoke and successful and successful % 384 == 0 and not result['overflow']:
                report['checkpoints'].append(save_checkpoint(args.out_dir / f'checkpoint_{successful:04d}.pt',
                    bank, optimizer, scaler, binding, successful, attempts))
            if attempts % 16 == 0 or successful == limit or result['overflow']:
                put(args.out_dir / 'report.json', report)
                print(json.dumps(dict(updates=successful, attempts=attempts,
                                      mk_ce=result['mk_ce'], prose_ce=result['prose_ce'],
                                      overflow=result['overflow'])), flush=True)
        need(bank.assert_base_frozen(check_values=True)['actual_content_checked'] is True,
             'Student frozen base content check failed')
        need(native.tensor_hash(student_execution.permutations) == parent[7]['table_sha256'],
             'Training state table changed')
        need(check_model(teacher, expected)['actual_content_checked'] is True,
             'Teacher frozen base content check failed')
        check_optimizer(optimizer.state_dict(), bank.state_dict(), successful,
                        lr_factor(successful - 1))
        adapter_path = args.out_dir / ('discarded_smoke_adapter_fp16.pt' if args.smoke else 'adapter_fp16.pt')
        exported = bank.export_fp16(adapter_path, binding=binding)
        payload = native.read_fp16(adapter_path, expected_binding=binding)
        need(sum(v.numel() * v.element_size() for v in payload['tensors'].values()) == 2308208,
             'FP16 adapter payload differs')
        need(any(bool(torch.count_nonzero(v)) for k, v in payload['tensors'].items()
                 if k.endswith('.V_read')), 'Exported adapter is identity')
        report['adapter'] = dict(exported, discarded=bool(args.smoke))
        with torch.no_grad():
            trained = {length: bank.forward_hidden(ids, use_checkpoint=False)[0].detach().clone()
                       for length, (ids, _, _) in reference.items()}
        need(student_execution.assert_static_values(), 'Static ridge values changed')
        static_hashes = dict(bases=[native.tensor_hash(x) for x in student_execution.bases],
            scales=[native.tensor_hash(x) for x in student_execution.scales],
            predictors=[[native.tensor_hash(x) for x in layer] for layer in student_execution.predictors])
        need(static_hashes == parent[7]['static_tensor_sha256'], 'Static ridge final hashes differ')
        report['frozen_static_ridge_sha256'] = static_hashes
        student_execution.close(); student_execution = None
        bank.close(); bank = None
        exported_raw = {}
        with torch.no_grad(), native.install_fp16(student, adapter_path,
                                                   expected_binding=binding,
                                                   expected_base_hashes=expected):
            with PredictorState(student, *parent[:5]) as packed:
                for length, (ids, _, old_cache) in reference.items():
                    actual = packed.backbone(ids, reset=True)
                    need(torch.equal(actual, trained[length])
                         and packed.cache_breakdown() == old_cache,
                         f'Exported packed/training parity failed: {length}')
                    exported_raw[str(length)] = dict(ids=ids.cpu(), packed_hidden=actual.cpu(), training_hidden=trained[length].cpu())
        report['parity_128_512_initial_and_export'] = True
        evidence_path = args.out_dir/'parity_evidence.pt'
        torch.save(dict(format='FP4_RIDGE_RESURFACE_PARITY_V3',binding=binding,initial=initial_raw,export=exported_raw),evidence_path)
        report['parity_evidence'] = dict(file=evidence_path.name,sha256=fp4.sha(evidence_path),bytes=evidence_path.stat().st_size)
        report['final_student_base_check'] = check_model(student, expected)
        report['final_teacher_base_check'] = check_model(teacher, expected)
        report['backend_final_check'] = check_replay_backend(policy)
        need(hashes() == report['code_sha256'], 'Training code changed during run')
        report.update(complete=True, finished_unix=time.time())
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc(), finished_unix=time.time())
        raise
    finally:
        for resource in (student_execution, teacher_execution, bank):
            if resource is not None:
                resource.close()
        put(args.out_dir / 'report.json', report)


if __name__ == '__main__':
    main()
