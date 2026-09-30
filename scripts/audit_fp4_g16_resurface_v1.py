#!/usr/bin/env python3
"""CPU-only arithmetic, provenance and paired-case audit of FP4 G16 Resurface."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def check_ppl(row):
    values = row['ppl']['windows']
    need(row['complete'] is True and row['ppl_complete'] is True
         and len(values) == 130 and sum(v['target_tokens'] for v in values) == 264764,
         'Full PPL population differs')
    total = 0.
    for item in values:
        need(math.isfinite(item['nll']) and item['nll'] >= 0
             and item['ppl'] == math.exp(item['nll'] / item['target_tokens']),
             'Window NLL/PPL arithmetic differs')
        total += item['nll']
    need(row['ppl']['nll'] == total and row['ppl']['target_tokens'] == 264764
         and row['ppl']['ppl'] == math.exp(total / 264764),
         'Aggregate NLL/PPL arithmetic differs')
    return row['ppl']['ppl']


def check_mk(row):
    rows = row['mk']['rows']
    normal = [r for r in rows if r['condition'] == 'normal']
    removed = [r for r in rows if r['condition'] == 'target_removed']
    need(len(rows) == 768 and len(normal) == len(removed) == 384
         and len({r['id'] for r in rows}) == 768, 'Full MK population differs')
    for condition, entries in (('normal', normal), ('target_removed', removed)):
        summary = row['mk']['summary'][condition]
        need(summary['count'] == len(entries)
             and summary['correct'] == sum(r['prediction'] == r['answer'] for r in entries)
             and summary['accuracy'] == summary['correct'] / len(entries),
             'MK summary arithmetic differs')
        for item in entries:
            need(item['correct'] == (item['prediction'] == item['answer'])
                 and len(item['generated_ids']) <= 12,
                 'MK exact-match/generation accounting differs')
    return {r['id']: r for r in normal}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--training-report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    need(not args.out.exists(), 'Fresh CPU audit output required')
    report = read(args.comparison)
    need(report['format'] == 'FP4_G16_RESURFACE_EVAL_V1'
         and report['complete'] is True, 'Complete paired result required')
    train = read(args.training_report)
    need(sha(args.training_report) == report['training_report_sha256']
         and train['format'] == 'FP4_G16_RESURFACE_TRAIN_V1'
         and train['complete'] is True and train['mode'] == 'formal'
         and train['successful_updates'] == 1536
         and train['attempts'] == len(train['history'])
         and train['overflows'] <= 8
         and train['attempts'] - train['overflows'] == 1536
         and train['fresh_initialization'] is True
         and train['parity_128_512_initial_and_export'] is True,
         'Final fresh training evidence differs')
    for path, digest in train['code_sha256'].items():
        source = (ROOT / path).resolve()
        need(not Path(path).is_absolute() and source.is_relative_to(ROOT.resolve())
             and source.is_file() and sha(source) == digest,
             'Training source identity differs: ' + path)
    prior = read(ROOT / 'artifacts/fp4_weight_v1/fp4_g16_v1/conversion_receipt.json')
    expected = {name: value['decoded_sha256'] for name, value in prior['tensors'].items()}
    for field in ('initial_student_base_check', 'initial_teacher_base_check',
                  'final_student_base_check', 'final_teacher_base_check'):
        value = train[field]
        need(value['actual_content_checked'] is True and value['tensors'] == 507
             and value['decoded_tensor_sha256'] == expected,
             'Training weight content differs: ' + field)
    need(len(train['checkpoints']) == 4
         and [c['successful_updates'] for c in train['checkpoints']] == [384, 768, 1152, 1536],
         'Formal checkpoint sequence differs')
    for checkpoint in train['checkpoints']:
        path = args.training_report.parent / checkpoint['file']
        need(path.is_file() and sha(path) == checkpoint['sha256']
             and path.stat().st_size == checkpoint['bytes'],
             'Checkpoint file differs')
    adapter_path = args.training_report.parent / train['adapter']['file']
    need(adapter_path.is_file() and sha(adapter_path) == train['adapter']['sha256']
         and adapter_path.stat().st_size == train['adapter']['bytes']
         and train['adapter']['payload_bytes'] == 2308208
         and train['adapter']['discarded'] is False,
         'Final adapter export differs')
    smoke = args.training_report.parent.parent / 'smoke' / 'report.json'
    need(sha(smoke) == train['smoke_report_sha256']
         and read(smoke)['adapter']['discarded'] is True,
         'Distinct discarded smoke evidence differs')
    directory = args.comparison.parent
    arms = ('fp4_g16_sq325', 'fp4_g16_sq325_resurface')
    rows = {}
    for arm in arms:
        entry = report['reports'][arm]
        path = directory / entry['file']
        need(path.parent == directory and sha(path) == entry['sha256'],
             'Bound arm file/hash differs')
        rows[arm] = read(path)
        need(rows[arm]['arm'] == arm and rows[arm]['weight_format'] == 'fp4_g16_e4m3'
             and rows[arm]['candidate_table_sha256'] ==
             '214b47a4dfdc20fce4aa552f954e3f0af84b49edbfc14cbcc1569946a4777ef8'
             and rows[arm]['frozen_weight_check']['actual_content_checked'] is True,
             'Arm source/table/frozen-base evidence differs')
        check_ppl(rows[arm])
    control, adapted = (rows[a] for a in arms)
    baseline = read(ROOT / 'artifacts/fp4_weight_v1/fp4_g16_v1/full_sq325.json')
    need(control['ppl']['windows'] == baseline['ppl']['windows']
         and control['ppl']['ppl'] == baseline['ppl']['ppl'] == 8.569404321843175,
         'Unadapted PPL does not exactly replay archived baseline')
    normal0, normal1 = check_mk(control), check_mk(adapted)
    need(set(normal0) == set(normal1), 'Paired normal MK IDs differ')
    changes = []
    for key in sorted(normal0):
        a, b = normal0[key], normal1[key]
        need(all(a[k] == b[k] for k in ('prompt_token_sha256_int64le', 'answer', 'condition')),
             'Paired MK prompt/answer changed')
        changes.append(int(b['correct']) - int(a['correct']))
    pair = report['comparison']
    need(pair['normal_mk_correct_delta'] == sum(changes)
         and pair['paired_improvements'] == sum(x > 0 for x in changes)
         and pair['paired_regressions'] == sum(x < 0 for x in changes)
         and pair['paired_unchanged'] == sum(x == 0 for x in changes)
         and pair['control_ppl'] == control['ppl']['ppl']
         and pair['candidate_ppl'] == adapted['ppl']['ppl'],
         'Paired comparison arithmetic differs')
    need(control['adapter_loaded'] is False and adapted['adapter_loaded'] is True
         and control['mk_cache']['total_bytes'] == adapted['mk_cache']['total_bytes'] == 28499968
         and adapted['adapter_storage']['resident_storage_bytes'] == 2308208
         and report['memory']['encoded_weight_payload_bytes'] == 4638460360,
         'Adapter/cache/encoded-payload accounting differs')
    check = dict(format='FP4_G16_RESURFACE_CPU_AUDIT_V1', complete=True, passed=True,
                 cuda_initialized=False, input_report_sha256=sha(args.comparison),
                 training_report_sha256=sha(args.training_report),
                 source_sha256=sha(__file__), full_ppl_windows=130, full_ppl_targets=264764,
                 full_mk_cases_per_arm=768, normal_mk_cases_per_arm=384,
                 unadapted_ppl=control['ppl']['ppl'], adapted_ppl=adapted['ppl']['ppl'],
                 normal_mk_delta=sum(changes), quality_gate_pass=report['quality_gate_pass'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(check, indent=2) + '\n')
    print(json.dumps(check, indent=2))


if __name__ == '__main__':
    main()
