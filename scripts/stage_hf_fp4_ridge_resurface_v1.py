#!/usr/bin/env python3
"""Stage the complete FP4/SQ3.25 bundle after verified GitHub publication.

Local filesystem only: no authentication, repository creation or upload. Keeps
the complete GitHub bundle unchanged under release/ and adds a reviewed HF card.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_fp4_ridge_resurface_v1 import verify_bundle, need, NAME, TAG, GITHUB_REPO, HF_REPO
from mamba2_recall.fp4_checkpoint import sha_file

FORMAT = 'MAMBA2_FP4_G16_SQ325_HF_PAYLOAD_V1'
ATTRIBUTES = b'*.safetensors filter=lfs diff=lfs merge=lfs -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n*.model filter=lfs diff=lfs merge=lfs -text\n'


def read(path):
    return json.loads(Path(path).read_text())


def prepare(args):
    bundle = Path(args.bundle)
    manifest = verify_bundle(bundle)
    built = read(args.build_receipt)
    github = read(args.github_receipt)
    need(built['complete'] is True and built['verified'] is True
         and built['bundle_manifest_sha256'] == sha_file(bundle / 'manifest.json')
         and built['github_repo'] == GITHUB_REPO and built['hugging_face_repo'] == HF_REPO
         and built['release_tag'] == TAG, 'Exact conditional GitHub bundle build is required')
    need(github['complete'] is True and github['verified'] is True
         and github['anonymous_access_verified'] is True
         and github['repository'] == GITHUB_REPO and github['tag'] == TAG
         and re.fullmatch(r'[0-9a-f]{40}', github['commit'])
         and github['bundle_manifest_sha256'] == built['bundle_manifest_sha256'],
         'Verified prior public GitHub release receipt is required before HF staging')
    expected_assets = {Path(item['path']).name: {'sha256': item['sha256'], 'bytes': item['bytes']}
                       for item in built['github_assets']}
    need(len(expected_assets) == len(built['github_assets'])
         and github['assets'] == expected_assets, 'Published GitHub asset inventory/digests differ')
    card = Path(args.card).read_bytes()
    need(card.startswith(b'---\n') and HF_REPO.encode() in card,
         'Reviewed HF model card must name the exact new repository')
    need(re.search(rb'\{\{[^{}\n]{1,120}\}\}', card) is None and b'DRAFT' not in card,
         'Draft or unresolved model-card placeholders must be replaced before staging')
    license_notice = Path(args.licenses).read_bytes()
    need(b'Apache' in license_notice and b'GPL' in license_notice,
         'Separate upstream weight and project/adapter license notices are required')
    stage = Path(args.stage)
    need(not stage.exists() and not stage.is_symlink(), 'Fresh HF staging directory required')
    stage.mkdir(parents=True)
    shutil.copytree(bundle, stage / 'release')
    (stage / 'README.md').write_bytes(card)
    (stage / 'LICENSES.md').write_bytes(license_notice)
    (stage / '.gitattributes').write_bytes(ATTRIBUTES)
    # Recheck exact bundle bytes after the local copy. No source-code or measured
    # documentation rewrite is performed; the HF model card supplies root links.
    need(verify_bundle(stage / 'release') == manifest, 'HF copied bundle differs')
    weight_names = set(read(bundle / 'weights/weight_manifest.json')['files'])
    files = {}
    for path in sorted(stage.rglob('*')):
        need(not path.is_symlink(), 'HF payload symlink')
        if not path.is_file():
            continue
        name = str(path.relative_to(stage))
        item = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
        if name.startswith('release/'):
            original = name.removeprefix('release/')
            if original.startswith('weights/') and original.removeprefix('weights/') in weight_names:
                item.update(source='verified_github_weight_asset', asset_name=Path(original).name)
            else:
                item.update(source='verified_github_runtime_archive', archive_name=NAME + '-runtime.tar.gz',
                            archive_member=NAME + '/' + original)
            item.update(github_repository=GITHUB_REPO, github_tag=TAG, github_commit=github['commit'])
        else:
            item['source'] = 'reviewed_hf_card' if name == 'README.md' else 'license_notice' if name == 'LICENSES.md' else 'lfs_attributes'
        files[name] = item
    publish = dict(format=FORMAT, complete=True, ready_for_upload=True, repository=HF_REPO,
        github_repository=GITHUB_REPO, github_tag=TAG, github_commit=github['commit'],
        github_receipt_sha256=sha_file(args.github_receipt),
        github_verification_scope=github.get('verification_scope', 'Scope recorded by supplied verified publication receipt'),
        bundle_manifest_sha256=sha_file(bundle / 'manifest.json'),
        candidate_ppl=manifest['candidate_ppl'], baseline_ppl=manifest['baseline_ppl'],
        file_count_excluding_root_manifest_checksums=len(files),
        file_bytes_excluding_root_manifest_checksums=sum(item['bytes'] for item in files.values()),
        files=files, new_quality_evaluation_performed=False,
        scope='Byte-identical complete GitHub packed-weight/runtime/state/adapter bundle, plus reviewed HF presentation')
    (stage / 'publish_manifest.json').write_text(json.dumps(publish, indent=2, sort_keys=True) + '\n')
    names = sorted([*files, 'publish_manifest.json'])
    (stage / 'SHA256SUMS').write_text(''.join(f'{sha_file(stage / name)}  {name}\n' for name in names))
    receipt = dict(format=FORMAT, complete=True, ready_for_upload=True, upload_performed=False,
                   repository=HF_REPO, github_commit=github['commit'], file_count=len(files) + 2,
                   total_file_bytes=sum(path.stat().st_size for path in stage.rglob('*') if path.is_file()),
                   publish_manifest_sha256=sha_file(stage / 'publish_manifest.json'),
                   checksums_sha256=sha_file(stage / 'SHA256SUMS'))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'build-receipt', 'github-receipt', 'card', 'licenses', 'stage'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == '__main__':
    main()
