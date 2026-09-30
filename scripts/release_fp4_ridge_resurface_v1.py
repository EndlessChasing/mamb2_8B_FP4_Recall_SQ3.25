#!/usr/bin/env python3
"""Build/verify a conditional FP4 G16 + SQ3.25 ridge Resurface public bundle.

Local files only. Requires audited full PPL <8, full MK, exact parent replay,
adapter removal and a source-free packed checkpoint reload. Large safetensors
shards are separate GitHub assets; the deterministic runtime tar excludes them.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tarfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall.fp4_checkpoint import verify_checkpoint, sha_file, need, LEDGER_SHA, SOURCE_SHA

FORMAT = 'MAMBA2_FP4_G16_SQ325_RIDGE_RESURFACE_RELEASE_V1'
STATE_FORMAT = 'MAMBA2_FP4_G16_SQ325_RIDGE_STATE_V1'
TAG = 'v0.1.0-fp4g16-sq325-resurface'
NAME = 'mamba2-8b-fp4g16-sq325-resurface-v1'
GITHUB_REPO = 'EndlessChasing/mamb2_8B_FP4_Recall_SQ3.25'
HF_REPO = 'EndlessChasing/Mamb2_8B_FP4_Recall_SQ3.25'
DIAGNOSTIC_DOC = 'docs/FP4_G16_FP_STATE_FINAL_DIAGNOSIS.md'
DIAGNOSTIC_DIR = 'docs/evidence/fp_state_diagnostics'
# Supplemental diagnostic evidence only. No raw corpus/tokens, state snapshots,
# training tensors or broad artifact-directory copy is allowed.
DIAGNOSTIC_SOURCES = {
    'v12/comparison.json': 'artifacts/fp4_zero_predictor_v1/fp_state_screen_v12c/comparison.json',
    'v12/audit.json': 'artifacts/fp4_zero_predictor_v1/fp_state_screen_v12c/audit.json',
    'v12/codec_fixture.json': 'artifacts/fp4_zero_predictor_v1/fp_state_codec_check_v12.json',
    'v13/endpoints_report.json': 'artifacts/fp4_zero_predictor_v1/fp_state_diag_v13_endpoints/report.json',
    'v13/endpoints_audit.json': 'artifacts/fp4_zero_predictor_v1/fp_state_diag_v13_endpoints/audit.json',
    'v13/production_oracle.json': 'artifacts/fp4_zero_predictor_v1/fp_state_diag_v13_oracle/report.json',
    'v13/cpu_reference.json': 'artifacts/fp4_zero_predictor_v1/fp_state_diag_v13_cpu/report.json',
    'v14/comparison.json': 'artifacts/fp4_zero_predictor_v1/selective_screen_v14/comparison.json',
    'v14/audit.json': 'artifacts/fp4_zero_predictor_v1/selective_screen_v14/audit.json',
    'v14/codec_fixture.json': 'artifacts/fp4_zero_predictor_v1/selective_codec_check_v14.json',
    'parent_full_audit.json': 'artifacts/fp4_zero_predictor_v1/group_ridge_full_v1/audit.json',
}
DIAGNOSTIC_PUBLIC_FILES = {DIAGNOSTIC_DIR + '/' + name for name in DIAGNOSTIC_SOURCES} | {
    DIAGNOSTIC_DIR + '/inventory.json', DIAGNOSTIC_DIR + '/README.md', DIAGNOSTIC_DOC,
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def relative(name):
    path = Path(name)
    need(isinstance(name, str) and not path.is_absolute() and '..' not in path.parts
         and str(path) == name, 'Unsafe relative publication path')
    return path


def local_imports(path):
    """Resolve only imports that have source files in this repository."""
    tree = ast.parse((ROOT / path).read_text())
    candidates = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = Path(path).parent
                for _ in range(node.level - 1):
                    base = base.parent
                module = base / ((node.module or '').replace('.', '/'))
                names.append(str(module).replace('/', '.'))
                names.extend(str(module / alias.name).replace('/', '.') for alias in node.names)
            else:
                names.append(node.module or '')
                names.extend((node.module + '.' if node.module else '') + alias.name for alias in node.names)
        for name in names:
            if not name or '*' in name:
                continue
            stem = name.replace('.', '/')
            for filename in (stem + '.py', stem + '/__init__.py', 'scripts/' + stem + '.py'):
                if (ROOT / filename).is_file():
                    candidates.add(filename)
    return candidates


def source_closure(seed):
    result = set(seed)
    queue = list(seed)
    while queue:
        name = queue.pop()
        relative(name)
        path = ROOT / name
        need(path.is_file() and not path.is_symlink(), 'Missing regular source file: ' + name)
        if name.endswith('.py'):
            for child in local_imports(name) - result:
                result.add(child)
                queue.append(child)
    return result


def diagnostic_evidence_sources():
    """Resolve the bounded explicit list; checks payload safety, not model quality."""
    result = {}
    total = 0
    forbidden = {'input_ids', 'token_ids', 'raw_tokens', 'raw_text', 'token_stream',
                 'snapshot_tensors', 'state_tensor', 'ssm_state', 'cached_tensor', 'state_values'}

    def no_raw_payload(value):
        if isinstance(value, dict):
            need(not forbidden.intersection(value), 'Raw token/state payload in diagnostic summary')
            for child in value.values():
                no_raw_payload(child)
        elif isinstance(value, list):
            need(len(value) <= 4096, 'Unbounded diagnostic list cannot be included')
            for child in value:
                no_raw_payload(child)

    for name, original in DIAGNOSTIC_SOURCES.items():
        relative(name)
        path = ROOT / relative(original)
        need(path.is_file() and not path.is_symlink(), 'Missing regular diagnostic summary: ' + original)
        size = path.stat().st_size
        need(0 < size <= 2 * 1024**2, 'Diagnostic summary exceeds 2 MiB: ' + original)
        total += size
        need(total <= 4 * 1024**2, 'Diagnostic summary inventory exceeds 4 MiB')
        report = read(path)
        no_raw_payload(report)
        result[name] = {'source': original, 'sha256': sha_file(path), 'bytes': size,
                        'format': report.get('format')}
    return result


def add_diagnostic_evidence(bundle, records):
    """Copy original JSON verbatim and fix only links in the packaged diagnosis doc."""
    directory = Path(bundle) / DIAGNOSTIC_DIR
    directory.mkdir(parents=True)
    for name, item in records.items():
        source = ROOT / item['source']
        need(source.stat().st_size == item['bytes'] and sha_file(source) == item['sha256'],
             'Diagnostic evidence changed during bundle preparation')
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    original_doc = ROOT / DIAGNOSTIC_DOC
    text = original_doc.read_text()
    replacements = {
        '../artifacts/fp4_zero_predictor_v1/fp_state_screen_v12c/audit.json':
            'evidence/fp_state_diagnostics/v12/audit.json',
        '../artifacts/fp4_zero_predictor_v1/fp_state_diag_v13_endpoints/audit.json':
            'evidence/fp_state_diagnostics/v13/endpoints_audit.json',
        '../artifacts/fp4_zero_predictor_v1/selective_screen_v14/audit.json':
            'evidence/fp_state_diagnostics/v14/audit.json',
        '../artifacts/fp4_zero_predictor_v1/group_ridge_full_v1/audit.json':
            'evidence/fp_state_diagnostics/parent_full_audit.json',
    }
    for original, target in replacements.items():
        text = text.replace(original, target)
    (Path(bundle) / DIAGNOSTIC_DOC).write_text(text)
    (directory / 'README.md').write_text(
        '# Supplemental FP8/FP4 state evidence\n\n'
        'These original JSON summaries, CPU audits and constructed codec fixtures document '
        'the bounded state-format investigation. They do not add a publication quality gate '
        'and do not claim that every possible floating-format recipe is ineffective.\n\n'
        '- `v12/`: matched TRAIN screening comparison, CPU audit and 28 codec/layout fixtures.\n'
        '- `v13/`: endpoint metric report/audit, independent production-storage oracle '
        'and CPU reference checks. The endpoint `snapshots` field contains only row, '
        'endpoint and sample-count metadata; no state tensors are included.\n'
        '- `v14/`: disjoint-window TRAIN screening comparison, CPU audit, explicit static '
        'layer selection policy and selective-codec fixtures.\n'
        '- `parent_full_audit.json`: historical full unadapted group-ridge parent audit.\n\n'
        'Per-arm source reports are referenced by their original summary/audit hashes; '
        'raw training token streams and state snapshots are omitted. '
        '`inventory.json` records original paths, bytes and SHA-256 values. '
        'All copied JSON bytes are unchanged; only the packaged diagnosis document\'s '
        'links are adjusted to point to this included evidence.\n')
    inventory = dict(format='MAMBA2_FP_STATE_DIAGNOSTIC_EVIDENCE_V1', complete=True,
                     supplemental_only=True, publication_quality_gate_unchanged=True,
                     files=records, original_diagnosis_sha256=sha_file(original_doc),
                     packaged_diagnosis_sha256=sha_file(Path(bundle) / DIAGNOSTIC_DOC),
                     diagnosis_link_replacements=replacements,
                     raw_training_tokens_included=False, state_snapshot_tensors_included=False)
    write(directory / 'inventory.json', inventory)
    return {'supplemental_only': True, 'publication_quality_gate_unchanged': True,
            'directory': DIAGNOSTIC_DIR, 'diagnosis_document': DIAGNOSTIC_DOC,
            'original_json_files': len(records),
            'original_json_bytes': sum(item['bytes'] for item in records.values()),
            'inventory_sha256': sha_file(directory / 'inventory.json')}


def quality(directory):
    """Check independently audited evidence bindings; does not rerun GPU quality."""
    directory = Path(directory)
    comp, audit, train, ta = (read(directory / 'evidence' / name) for name in
                            ('comparison.json', 'audit.json', 'training_report.json', 'training_audit.json'))
    need(comp['complete'] is True and comp['stage'] == 'full'
         and comp['mk_complete'] is True and comp['publication_gate_pass'] is True
         and comp['strict_ppl_below_8'] is True and comp['normal_mk_gate_pass'] is True,
         'Full strict PPL/MK publication gate must pass')
    need(audit['complete'] is True and audit['passed'] is True and audit['stage'] == 'full'
         and audit['cuda_initialized'] is False and audit['publication_gate_pass'] is True
         and audit['input_report_sha256'] == sha_file(directory / 'evidence/comparison.json')
         and audit['training_report_sha256'] == sha_file(directory / 'evidence/training_report.json'),
         'Complete bound independent CPU audit is required')
    need(comp['training_report_sha256'] == sha_file(directory / 'evidence/training_report.json')
         and comp['training_audit_sha256'] == sha_file(directory / 'evidence/training_audit.json')
         and train['complete'] is True and ta['complete'] is True and ta['passed'] is True
         and ta['cuda_initialized'] is False
         and ta['training_report_sha256'] == sha_file(directory / 'evidence/training_report.json'),
         'Training/audit binding differs')
    need(comp['parent']['parent_ppl'] == 8.408282583578627
         and comp['exact_parent_replay']['complete'] is True
         and comp['adapter_removal_reset_and_cache_exact'] is True,
         'Exact original ridge parent and adapter removal are required')
    for name in ('ridge_parent', 'ridge_resurface'):
        item = comp['reports'][name]
        row = read(directory / 'evidence' / (name + '.json'))
        need(sha_file(directory / 'evidence' / (name + '.json')) == item['sha256']
             and row['complete'] is True and row['ppl_complete'] is True and row['mk_complete'] is True
             and len(row['ppl']['windows']) == 130 and row['ppl']['target_tokens'] == 264764
             and len(row['mk']['rows']) == 768
             and row['ppl']['ppl'] == comp['ppl'][name] == item['ppl'],
             'Full paired arm/hash/population differs: ' + name)
        nll = 0.
        for window in row['ppl']['windows']:
            nll += window['nll']
        need(math.isfinite(nll) and nll == row['ppl']['nll']
             and math.exp(nll / 264764) == row['ppl']['ppl'], 'PPL arithmetic differs')
    need(comp['ppl']['ridge_parent'] == 8.408282583578627
         and math.isfinite(comp['ppl']['ridge_resurface']) and comp['ppl']['ridge_resurface'] < 8.,
         'Adapted full PPL must be strictly below 8.0')
    need(comp['adapter_sha256'] == train['adapter']['sha256'] == sha_file(directory / 'adapter_fp16.pt')
         and train['adapter']['payload_bytes'] == 2308208
         and train['adapter']['bytes'] == (directory / 'adapter_fp16.pt').stat().st_size,
         'Final adapter file/payload differs')
    for evidence in (comp, train):
        for name, digest in evidence['code_sha256'].items():
            relative(name)
            need(sha_file(directory / name) == digest, 'Measured source differs: ' + name)
    reload = read(directory / 'evidence/packed_reload.json')
    need(reload['complete'] is True and reload['passed'] is True and reload['source_checkpoint_required'] is False
         and reload['decoded_tensor_count'] == 507
         and reload['resident_fp16_weight_bytes'] == 16473999360
         and reload['weight_manifest_sha256'] == sha_file(directory / 'weights/weight_manifest.json')
         and reload['conversion_receipt_sha256'] == LEDGER_SHA,
         'Full source-free packed checkpoint reload is required')
    ledger = read(directory / 'weights/conversion_receipt.json')
    expected_hashes = {name: item['decoded_sha256'] for name, item in ledger['tensors'].items()}
    need({row['tensor']: row['decoded_sha256'] for row in reload['decoded_hash_records']} == expected_hashes,
         'Reloaded tensor hashes differ from quality ledger')
    for field in ('initial_weight_check', 'final_weight_check'):
        need(comp[field]['actual_content_checked'] is True
             and comp[field]['decoded_tensor_sha256'] == expected_hashes,
             'Quality run does not bind all 507 exact exported weight tensors')
    return comp, train


def verify_bundle(directory):
    """Verify complete public bundle with no torch, CUDA or source weights."""
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink()
         and not any(path.is_symlink() for path in directory.rglob('*')), 'No symlinks allowed')
    manifest = read(directory / 'manifest.json')
    need(manifest['format'] == FORMAT and manifest['complete'] is True
         and manifest['github_repo'] == GITHUB_REPO and manifest['hugging_face_repo'] == HF_REPO
         and manifest['release_tag'] == TAG, 'Wrong release identity')
    files = manifest['files']
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()}
    need(actual == set(files) | {'manifest.json', 'SHA256SUMS'}, 'Unexpected/missing publication files')
    hashes = {}
    for name, item in files.items():
        relative(name)
        path = directory / name
        hashes[name] = sha_file(path)
        need(item == {'sha256': hashes[name], 'bytes': path.stat().st_size}, 'Bundle file differs: ' + name)
    hashes['manifest.json'] = sha_file(directory / 'manifest.json')
    need((directory / 'SHA256SUMS').read_text() == ''.join(f'{hashes[name]}  {name}\n' for name in sorted(hashes)),
         'Bundle checksum list differs')
    comp, train = quality(directory)
    weights, _ = verify_checkpoint(directory / 'weights', code_root=directory)
    need(manifest['adapter_binding'] == train['binding'] and manifest['state_provenance'] == comp['parent']
         and manifest['backend_policy'] == comp['backend_policy']
         and manifest['candidate_ppl'] == comp['ppl']['ridge_resurface']
         and manifest['baseline_ppl'] == comp['ppl']['ridge_parent'], 'Manifest quality/configuration binding differs')
    memory = comp['memory']
    persistent = sum(memory[key] for key in ('state_conv_table_cache_bytes', 'static_basis_bytes',
                                            'static_latent_scale_bytes', 'static_predictor_bytes'))
    need(manifest['persistent_state_side_bytes'] == persistent == 32284928
         and manifest['state_plus_adapter_bytes'] == persistent + train['adapter']['payload_bytes']
         and manifest['packed_weight_payload_bytes'] == weights['logical_weight_payload_bytes'],
         'Persistent state/adapter/weight accounting differs')
    return manifest


def build(args):
    import torch
    from mamba2_recall import resurface_native as native, runtime
    sys.path.insert(0, str(ROOT / 'scripts'))
    import ridge_resurface_binding_v2 as shared
    output = Path(args.build)
    need(not output.exists() and not output.is_symlink(), 'Fresh release output directory required')
    comp, train = read(args.comparison), read(args.training_report)
    # Fail before allocating/copying the large payload when the condition has not passed.
    need(comp['complete'] is True and comp['stage'] == 'full' and comp['publication_gate_pass'] is True
         and comp['ppl']['ridge_resurface'] < 8., 'Audited full PPL/MK gate required before building')
    readme = Path(args.readme).read_bytes()
    need(re.search(rb'\{\{[^{}\n]{1,120}\}\}', readme) is None and b'DRAFT' not in readme,
         'Draft release README or unresolved placeholders cannot be packaged')
    diagnostic_records = diagnostic_evidence_sources()
    weights, ledger = verify_checkpoint(args.packed_checkpoint)
    parent_args = SimpleNamespace(static_dir=args.state_static_dir, ridge_full_dir=args.ridge_full_dir,
                                  top4_rank_dir=args.state_table_dir, top4_full_dir=args.top4_full_dir)
    table, layouts, bases, scales, predictors, _, _, provenance = shared.load_parent(parent_args)
    need(provenance == comp['parent'], 'Release state parent differs from evaluated model')
    adapter = native.read_fp16(args.adapter, expected_binding=train['binding'])
    need(sha_file(args.adapter) == train['adapter']['sha256'], 'Selected final adapter differs')
    need({name: native.tensor_hash(value) for name, value in adapter['tensors'].items()}
         == train['adapter']['tensor_sha256'], 'All 224 adapter tensor hashes must match training export')
    seed = set(comp['code_sha256']) | set(train['code_sha256']) | set(ledger['code_sha256']) | set(weights['code_sha256'])
    seed.update(('mamba2_recall/__init__.py', 'scripts/release_fp4_ridge_resurface_v1.py',
                 'scripts/infer_fp4_ridge_resurface_v1.py', 'scripts/check_fp4_g16_checkpoint_reload_v1.py',
                 'scripts/verify_github_fp4_ridge_resurface_v1.py', 'scripts/prepare_github_fp4_source_v1.py',
                 'scripts/verify_hf_fp4_release_v1.py',
                 'scripts/stage_hf_fp4_ridge_resurface_v1.py', 'scripts/finalize_fp4_release_card_v1.py',
                 'scripts/fp4_zero_predictor_codec_v1.py', 'scripts/evaluate_resurface_more.py',
                 'docs/RESURFACE_MORE_BACKEND_REPLAY.md',
                 DIAGNOSTIC_DOC,
                 'pyproject.toml', 'LICENSE', 'reference/statequant/LICENSE',
                 'reference/statequant/selective_state_update_pairnib.py',
                 'reference/w4/WEIGHTS_LICENSE.txt'))
    sources = source_closure(seed)
    output.mkdir(parents=True)
    bundle = output / NAME
    bundle.mkdir()
    for name in sorted(sources):
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    diagnostic_descriptor = add_diagnostic_evidence(bundle, diagnostic_records)
    shutil.copytree(args.packed_checkpoint, bundle / 'weights')
    (bundle / 'evidence').mkdir()
    payload_files = {'evidence/comparison.json': args.comparison, 'evidence/audit.json': args.audit,
                     'evidence/training_report.json': args.training_report, 'evidence/training_audit.json': args.training_audit,
                     'evidence/packed_reload.json': args.packed_reload_report, 'adapter_fp16.pt': args.adapter,
                     'README.md': args.readme, 'THIRD_PARTY_NOTICES.md': args.notices,
                     'docs/THIRD_PARTY_NOTICES.md': args.notices}
    for name in ('ridge_parent', 'ridge_resurface'):
        filename = comp['reports'][name]['file']
        need(Path(filename).name == filename, 'Unsafe measured arm filename')
        payload_files['evidence/' + name + '.json'] = args.comparison.parent / filename
    tokenizer = Path(args.tokenizer)
    need(sha_file(tokenizer) == runtime.TOKENIZER_SHA256, 'Pinned tokenizer required')
    payload_files['tokenizer/' + runtime.TOKENIZER_FILENAME] = tokenizer
    for name, source in payload_files.items():
        need(Path(source).is_file() and not Path(source).is_symlink(), 'Regular publication payload required: ' + name)
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    state = dict(format=STATE_FORMAT, provenance=provenance, table=table.clone().contiguous(),
                 layouts=list(layouts), bases=[x.clone().contiguous() for x in bases],
                 scales=[x.clone().contiguous() for x in scales],
                 stackedpredictors=[x.clone().contiguous() for x in predictors])
    with (bundle / 'state_config.pt').open('xb') as stream:
        torch.save(state, stream)
    restored = torch.load(bundle / 'state_config.pt', map_location='cpu', weights_only=True)
    need(restored['provenance'] == provenance and native.tensor_hash(restored['table']) == provenance['table_sha256'],
         'Exported state table/provenance differs')
    for key in ('bases', 'scales', 'stackedpredictors'):
        need(len(restored[key]) == 56 and all(torch.equal(x, y) for x, y in zip(restored[key], state[key])),
             'State static serialization differs: ' + key)
    need(restored['layouts'] == list(layouts), 'Serialized layer layouts differ')
    state_payload = sum(value.numel() * value.element_size() for key in ('bases', 'scales', 'stackedpredictors')
                        for value in state[key]) + table.numel()
    need(state_payload == 3842304 and sum(x.numel() * x.element_size() for x in adapter['tensors'].values()) == 2308208,
         'Static state/adapter tensor bytes differ')
    files = {str(path.relative_to(bundle)): {'sha256': sha_file(path), 'bytes': path.stat().st_size}
             for path in sorted(bundle.rglob('*')) if path.is_file()}
    manifest = dict(format=FORMAT, complete=True, release_tag=TAG, github_repo=GITHUB_REPO,
        hugging_face_repo=HF_REPO, candidate_ppl=comp['ppl']['ridge_resurface'], baseline_ppl=8.408282583578627,
        adapter_binding=train['binding'], state_provenance=provenance, backend_policy=comp['backend_policy'],
        source_checkpoint_sha256=SOURCE_SHA, tokenizer_sha256=runtime.TOKENIZER_SHA256,
        state_config_format=STATE_FORMAT, state_config_payload_bytes=state_payload,
        persistent_state_side_bytes=32284928, adapter_fp16_payload_bytes=2308208,
        state_plus_adapter_bytes=34593136, packed_weight_payload_bytes=weights['logical_weight_payload_bytes'],
        decoded_resident_weight_bytes=16473999360, packed_resident_kernel=False,
        fp_state_diagnostic_evidence=diagnostic_descriptor,
        files=files, measured_quality_scope=comp.get('quality_scope', 'Historically exposed validation; no untouched test claim'),
        scope='Complete packed weight files, custom FP16 compute runtime, state configuration and Resurface adapter')
    write(bundle / 'manifest.json', manifest)
    names = sorted([*files, 'manifest.json'])
    (bundle / 'SHA256SUMS').write_text(''.join(f'{sha_file(bundle / name)}  {name}\n' for name in names))
    verify_bundle(bundle)
    # GitHub caps each individual release asset below 2 GiB. Ship actual weight
    # shards independently and archive only runtime/configuration/evidence bytes.
    archive = output / (NAME + '-runtime.tar.gz')
    shards = {str(Path('weights') / name) for name in weights['files']}
    with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name in sorted((set(names) | {'SHA256SUMS'}) - shards):
                path = bundle / name
                info = tar.gettarinfo(str(path), arcname=NAME + '/' + name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    assets = [{'path': archive.name, 'sha256': sha_file(archive), 'bytes': archive.stat().st_size}]
    assets.extend({'path': NAME + '/weights/' + name, **weights['files'][name]} for name in sorted(weights['files']))
    need(len(assets) <= 1000 and all(item['bytes'] < 2 * 1024**3 for item in assets), 'GitHub release asset limit exceeded')
    receipt = dict(format=FORMAT, complete=True, verified=True, publication_performed=False,
                   github_repo=GITHUB_REPO, hugging_face_repo=HF_REPO, release_tag=TAG,
                   candidate_ppl=manifest['candidate_ppl'], bundle_manifest_sha256=sha_file(bundle / 'manifest.json'),
                   github_assets=assets, file_count=len(files) + 2,
                   total_file_bytes=sum(path.stat().st_size for path in bundle.rglob('*') if path.is_file()))
    write(output / 'build_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build', type=Path)
    mode.add_argument('--verify', type=Path)
    for key in ('comparison', 'audit', 'training-report', 'training-audit', 'adapter',
                'packed-checkpoint', 'packed-reload-report', 'state-static-dir',
                'state-table-dir', 'ridge-full-dir', 'top4-full-dir', 'tokenizer', 'readme', 'notices'):
        parser.add_argument('--' + key, type=Path)
    args = parser.parse_args()
    if args.build:
        required = [key for key, value in vars(args).items() if key not in ('build', 'verify') and value is None]
        if required:
            parser.error('--build requires ' + ', '.join('--' + key.replace('_', '-') for key in required))
        result = build(args)
    else:
        manifest = verify_bundle(args.verify)
        result = {'verified': True, 'candidate_ppl': manifest['candidate_ppl'],
                  'baseline_ppl': manifest['baseline_ppl'], 'scope': manifest['scope']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
