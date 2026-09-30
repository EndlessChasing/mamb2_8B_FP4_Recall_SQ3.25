#!/usr/bin/env python3
"""Render a conditional model card only from complete, bound PPL/MK audits.

This helper creates local documentation. It does not publish or rerun quality.
The final release builder still verifies the entire payload independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PARENT_PPL = 8.408282583578627
EXPECTED_PACKAGES = {'torch': '2.11.0+cu128', 'mamba-ssm': '2.3.2.post1', 'triton': '3.6.0'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound_arm(comparison, comparison_path, name):
    item = comparison['reports'][name]
    filename = item['file']
    need(isinstance(filename, str) and Path(filename).name == filename, 'Unsafe arm filename')
    path = comparison_path.parent / filename
    need(sha_file(path) == item['sha256'], 'Arm report hash differs: ' + name)
    row = read(path)
    need(row['complete'] is True and row['ppl_complete'] is True and row['mk_complete'] is True,
         'Full arm is incomplete: ' + name)
    ppl = row['ppl']
    need(len(ppl['windows']) == 130 and ppl['target_tokens'] == 264764,
         'Full PPL population differs: ' + name)
    nll = sum(window['nll'] for window in ppl['windows'])
    need(nll == ppl['nll'] and sum(window['target_tokens'] for window in ppl['windows']) == 264764
         and math.isfinite(nll) and math.exp(nll / 264764) == ppl['ppl']
         and ppl['ppl'] == comparison['ppl'][name] == item['ppl'], 'PPL arithmetic differs: ' + name)
    mk = row['mk']
    need(len(mk['rows']) == 768, 'Full MK population differs: ' + name)
    for condition in ('normal', 'target_removed'):
        subset = [case for case in mk['rows'] if case['condition'] == condition]
        summary = mk['summary'][condition]
        need(len(subset) == summary['count'] == 384
             and sum(case['correct'] for case in subset) == summary['correct']
             and summary['accuracy'] == summary['correct'] / 384,
             'MK summary differs: ' + name + '/' + condition)
    return row


def render(args):
    comp, audit, train, training_audit = (read(path) for path in
                                        (args.comparison, args.audit, args.training_report, args.training_audit))
    need(comp['complete'] is True and comp['stage'] == 'full' and comp['mk_complete'] is True
         and comp['publication_gate_pass'] is True and comp['strict_ppl_below_8'] is True
         and comp['normal_mk_gate_pass'] is True, 'Complete passing full PPL/MK comparison required')
    need(audit['complete'] is True and audit['passed'] is True and audit['stage'] == 'full'
         and audit['cuda_initialized'] is False and audit['publication_gate_pass'] is True
         and audit['input_report_sha256'] == sha_file(args.comparison)
         and audit['training_report_sha256'] == sha_file(args.training_report),
         'Bound independent full CPU audit required')
    need(train['complete'] is True and training_audit['complete'] is True
         and training_audit['passed'] is True and training_audit['cuda_initialized'] is False
         and training_audit['training_report_sha256'] == sha_file(args.training_report)
         and comp['training_report_sha256'] == sha_file(args.training_report)
         and comp['training_audit_sha256'] == sha_file(args.training_audit),
         'Bound complete training and independent CPU audit required')
    need(comp['parent']['parent_ppl'] == PARENT_PPL and comp['exact_parent_replay']['complete'] is True
         and comp['adapter_removal_reset_and_cache_exact'] is True, 'Exact frozen parent/removal required')
    parent = bound_arm(comp, args.comparison, 'ridge_parent')
    candidate = bound_arm(comp, args.comparison, 'ridge_resurface')
    need(parent['ppl']['ppl'] == PARENT_PPL and 0 < candidate['ppl']['ppl'] < 8.,
         'Full PPL must be strictly below 8.0')
    pair = comp['comparison']
    controls = pair['control_normal_mk'], pair['candidate_normal_mk']
    need(controls == (parent['mk']['summary']['normal'], candidate['mk']['summary']['normal'])
         and pair['normal_mk_correct_delta'] == controls[1]['correct'] - controls[0]['correct'] > 0
         and pair['normal_mk_accuracy_delta'] == pair['normal_mk_correct_delta'] / 384,
         'Paired normal MK gain differs')
    ci = pair['normal_mk_paired_bootstrap_95ci']
    need(len(ci) == 2 and 0 < ci[0] <= ci[1] <= 1
         and pair['bootstrap_draws'] == 10000, 'Positive predeclared paired MK interval required')
    need(comp['backend_policy'] == train['backend_policy']
         and comp['backend_policy']['package_versions'] == EXPECTED_PACKAGES,
         'Recorded backend differs from the documented environment')
    need(train['adapter']['payload_bytes'] == 2308208 and train['adapter']['parameters'] == 1154104
         and comp['adapter_sha256'] == train['adapter']['sha256'], 'Adapter identity/payload differs')
    planned = train['binding']['planned_successful_updates']
    need(train['successful_updates'] == planned == 1536 and train.get('mode') == 'formal',
         'Final formal 1,536-update export required')
    origin = comp['adapter_origin']
    need(origin in ('fresh', 'transfer'), 'Unknown adapter origin')
    details = ('The released adapter was trained from fresh initialization on the exact '
               'frozen group-ridge state parent.' if origin == 'fresh' else
               'The released adapter was transferred from its earlier FP4 G16/SQ3.25 '
               'state parent and evaluated on the frozen group-ridge parent; it was '
               'not freshly trained on the group-ridge state codec.')
    details += (' Training used the pinned numeric TRAIN examples and WikiText TRAIN '
                'windows, with 1,536 successful updates; the final export was used '
                'without selecting a checkpoint on validation. The frozen FP4 G16 '
                'model with FP16 state supplied the prose teacher. See '
                '`evidence/training_report.json` and `evidence/training_audit.json` '
                'for the exact recipe, data hashes, export and frozen-tensor checks.')
    if origin == 'fresh':
        details += (' The state forward matches the deployed ridge codec. Training '
                    'uses a surrogate backward with a fixed live-mask straight-through '
                    'estimator and omits latent/predictor derivative terms; it does '
                    'not implement an exact adjoint of the codec.')
    if 'objective' in train['binding']:
        details += '\n\nThe frozen loss weights were `' + json.dumps(train['binding']['objective'], sort_keys=True) + '`.'
    values = {
        'PARENT_PPL': f"{parent['ppl']['ppl']:.8f}",
        'CANDIDATE_PPL': f"{candidate['ppl']['ppl']:.8f}",
        'PARENT_NORMAL_MK': str(controls[0]['correct']),
        'CANDIDATE_NORMAL_MK': str(controls[1]['correct']),
        'PARENT_REMOVED_MK': str(parent['mk']['summary']['target_removed']['correct']),
        'CANDIDATE_REMOVED_MK': str(candidate['mk']['summary']['target_removed']['correct']),
        'MK_GAIN_PP': f"{100 * pair['normal_mk_accuracy_delta']:.2f}",
        'MK_CI_LOW_PP': f'{100 * ci[0]:.2f}', 'MK_CI_HIGH_PP': f'{100 * ci[1]:.2f}',
        'BOOTSTRAP_DRAWS': f"{pair['bootstrap_draws']:,}",
        'ADAPTER_FILE_BYTES': f"{train['adapter']['bytes']:,}",
        'ADAPTER_SHA256': train['adapter']['sha256'], 'COMPARISON_SHA256': sha_file(args.comparison),
        'TORCH_VERSION': EXPECTED_PACKAGES['torch'], 'MAMBA_VERSION': EXPECTED_PACKAGES['mamba-ssm'],
        'TRITON_VERSION': EXPECTED_PACKAGES['triton'], 'TRAINING_DETAILS': details,
    }
    template = args.template.read_text()
    template = template.replace('<!-- CONDITIONAL_RELEASE_DRAFT: finalize only from complete, independently audited PPL/MK evidence. -->\n\n', '')
    for key, value in values.items():
        template = template.replace('{{' + key + '}}', value)
    need(not re.search(r'\{\{[^}]*\}\}', template) and 'CONDITIONAL_RELEASE_DRAFT' not in template,
         'Unresolved documentation placeholder')
    need(not args.out.exists(), 'Fresh model card output required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(template)
    return {'complete': True, 'documentation_only': True, 'publication_performed': False,
            'card_sha256': sha_file(args.out), 'comparison_sha256': sha_file(args.comparison),
            'candidate_ppl': candidate['ppl']['ppl'], 'normal_mk_gain': pair['normal_mk_correct_delta'],
            'adapter_origin': origin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison', 'audit', 'training-report', 'training-audit', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--template', type=Path, default=ROOT / 'docs/FP4_G16_RELEASE_README_DRAFT.md')
    args = parser.parse_args()
    print(json.dumps(render(args), indent=2))


if __name__ == '__main__':
    main()
