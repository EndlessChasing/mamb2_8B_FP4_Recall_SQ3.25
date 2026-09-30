#!/usr/bin/env python3
"""Create a fresh, allowlisted source directory for the conditional GitHub repo.

Local filesystem only. Does not initialize Git, commit, create a remote, or upload.
Never copies the research checkout history, weights, checkpoints, logs, or data.
The complete model remains in the separately verified release assets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
from release_fp4_ridge_resurface_v1 import (
    GITHUB_REPO, HF_REPO, NAME, TAG, DIAGNOSTIC_PUBLIC_FILES, need, quality, relative, sha_file,
)
from verify_github_fp4_ridge_resurface_v1 import expected_assets

FORMAT = 'MAMBA2_FP4_G16_SQ325_GITHUB_SOURCE_V1'
ROOT_FILES = {'README.md', 'THIRD_PARTY_NOTICES.md', 'LICENSE', 'pyproject.toml'}
IGNORE = b'__pycache__/\n*.pyc\n*.egg-info/\n.venv/\nartifacts/\nreports/\nlogs/\nweights/\n*.safetensors\n*.pt\n*.model\n*.log\n.DS_Store\n'


def source_allowed(name):
    path = relative(name)
    if name in ROOT_FILES:
        return True
    if path.parts[0] in ('mamba2_recall', 'scripts'):
        return path.suffix == '.py'
    if path.parts[0] == 'docs':
        return path.suffix == '.md' or name in DIAGNOSTIC_PUBLIC_FILES
    if path.parts[0] == 'reference':
        return path.suffix == '.py' or path.name in ('LICENSE', 'WEIGHTS_LICENSE.txt', 'WEIGHTS_NOTICE.md')
    return False


def prepare(args):
    bundle = args.bundle.resolve()
    need(not any(value.is_symlink() for value in (args.out, args.receipt, args.pathspec)),
         'Symlink source output paths are forbidden')
    args.out, args.receipt, args.pathspec = (value.resolve() for value in (args.out, args.receipt, args.pathspec))
    built = json.loads(args.build_receipt.read_text())
    assets = expected_assets(built)
    need(sha_file(bundle / 'manifest.json') == built['bundle_manifest_sha256'], 'Frozen build manifest differs')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    comp, train = quality(bundle)
    need(manifest['complete'] is True and manifest['github_repo'] == GITHUB_REPO
         and manifest['release_tag'] == TAG and manifest['candidate_ppl'] == built['candidate_ppl']
         and manifest['candidate_ppl'] == comp['ppl']['ridge_resurface']
         and manifest['adapter_binding'] == train['binding'], 'Passing exact source recipe required')
    need(not args.out.exists() and not args.out.is_symlink()
         and not args.receipt.exists() and not args.receipt.is_symlink()
         and not args.pathspec.exists() and not args.pathspec.is_symlink(), 'Fresh source, receipt and pathspec paths required')
    need(args.out not in args.receipt.parents and args.out not in args.pathspec.parents,
         'Source receipt/pathspec must be outside the new source checkout')
    need(bundle not in args.out.parents and bundle != args.out,
         'New source checkout must not change the frozen bundle')
    names = sorted(name for name in manifest['files'] if source_allowed(name))
    need(ROOT_FILES <= set(names) and 'scripts/infer_fp4_ridge_resurface_v1.py' in names,
         'Missing public README, notices, license or runtime source')
    selected = {}
    for name in names:
        path = bundle / relative(name)
        need(path.is_file() and not path.is_symlink(), 'Regular allowlisted source required')
        item = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
        need(item == manifest['files'][name], 'Allowlisted source differs: ' + name)
        if path.suffix == '.md':
            data = path.read_bytes()
            need(b'DRAFT' not in data and re.search(rb'\{\{[^{}\n]{1,120}\}\}', data) is None,
                 'Unresolved draft source documentation: ' + name)
        selected[name] = item
    args.out.mkdir(parents=True)
    for name in names:
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / name, target)
    (args.out / '.gitignore').write_bytes(IGNORE)
    provenance = dict(format=FORMAT, complete=True, repository=GITHUB_REPO, hugging_face_repo=HF_REPO,
        release_tag=TAG, bundle_name=NAME, bundle_manifest_sha256=built['bundle_manifest_sha256'],
        candidate_ppl=manifest['candidate_ppl'], baseline_ppl=manifest['baseline_ppl'],
        source_checkpoint_sha256=manifest['source_checkpoint_sha256'], tokenizer_sha256=manifest['tokenizer_sha256'],
        source_files=selected, release_assets=assets,
        supplemental_diagnostics=manifest.get('fp_state_diagnostic_evidence'),
        scope='Allowlisted source bytes and public asset provenance from the audited complete bundle; '
              'no research Git history, raw logs, checkpoints, tokenized corpus or weights included in source commit.')
    (args.out / 'PUBLICATION_PROVENANCE.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    names += ['.gitignore', 'PUBLICATION_PROVENANCE.json']
    args.pathspec.parent.mkdir(parents=True, exist_ok=True)
    with args.pathspec.open('xb') as stream:
        stream.write(b''.join(name.encode() + b'\0' for name in sorted(names)))
    final = {name: {'sha256': sha_file(args.out / name), 'bytes': (args.out / name).stat().st_size}
             for name in sorted(names)}
    receipt = dict(format=FORMAT, complete=True, verified=True, publication_performed=False,
        repository=GITHUB_REPO, tag=TAG, bundle_manifest_sha256=built['bundle_manifest_sha256'],
        source_directory=str(args.out.resolve()), files=final,
        pathspec_sha256=sha_file(args.pathspec), scope=provenance['scope'])
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'build-receipt', 'out', 'receipt', 'pathspec'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == '__main__':
    main()
