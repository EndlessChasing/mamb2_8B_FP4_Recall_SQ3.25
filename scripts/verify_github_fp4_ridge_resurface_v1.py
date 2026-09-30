#!/usr/bin/env python3
"""Verify the public FP4/SQ3.25 GitHub release using anonymous GET requests.

No credentials are read and no GitHub mutation is performed. The runtime archive
is downloaded and hashed in full. Every weight asset is anonymously sampled; its
SHA-256 is checked against GitHub's API digest. If an API digest is absent, that
asset is downloaded in full and hashed instead. Sampling is not a full file hash.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

sys.dont_write_bytecode = True
from release_fp4_ridge_resurface_v1 import (
    FORMAT as BUILD_FORMAT, GITHUB_REPO, HF_REPO, NAME, TAG, need, quality,
    relative, sha_file,
)

FORMAT = 'MAMBA2_FP4_G16_SQ325_GITHUB_PUBLICATION_CHECK_V1'
API = 'https://api.github.com'
USER_AGENT = 'Mamba2-FP4-SQ325-anonymous-release-verifier/1'
SHA = re.compile(r'[0-9a-f]{64}')
COMMIT = re.compile(r'[0-9a-f]{40}')
RUNTIME = NAME + '-runtime.tar.gz'
MAX_RUNTIME_UNPACKED = 512 * 1024**2
MAX_RUNTIME_MEMBER = 128 * 1024**2


def read(path):
    return json.loads(Path(path).read_text())


def write_fresh(path, value):
    path = Path(path)
    need(not path.exists() and not path.is_symlink(), 'Fresh receipt path required')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def expected_assets(built):
    need(built['format'] == BUILD_FORMAT and built['complete'] is True
         and built['verified'] is True and built['github_repo'] == GITHUB_REPO
         and built['hugging_face_repo'] == HF_REPO and built['release_tag'] == TAG
         and isinstance(built['candidate_ppl'], (int, float))
         and math.isfinite(built['candidate_ppl']) and 0 < built['candidate_ppl'] < 8
         and SHA.fullmatch(built['bundle_manifest_sha256']), 'Passing exact build receipt required')
    result = {}
    for item in built['github_assets']:
        path = relative(item['path'])
        name = path.name
        need(name not in result and SHA.fullmatch(item['sha256'])
             and type(item['bytes']) is int and 0 < item['bytes'] < 2 * 1024**3,
             'Invalid or duplicate build asset: ' + name)
        if name == RUNTIME:
            need(str(path) == name, 'Runtime asset must be at build root')
        else:
            need(path.parent == Path(NAME) / 'weights' and path.suffix == '.safetensors',
                 'Unexpected release asset: ' + name)
        result[name] = {'sha256': item['sha256'], 'bytes': item['bytes']}
    need(len(result) == 120 and RUNTIME in result
         and sum(name.endswith('.safetensors') for name in result) == 119,
         'Exactly one runtime archive and 119 weight shards required')
    return result


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        need(urlparse(newurl).scheme == 'https', 'Non-HTTPS download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class AnonymousHTTP:
    def __init__(self, timeout=30):
        self.timeout = timeout
        # No auth handlers, netrc, gh, hf, token access, or ambient authenticated
        # proxy. TLS certificate checks remain enabled.
        self.opener = build_opener(ProxyHandler({}), HTTPSRedirects())

    def open(self, url, *, headers=None):
        need(urlparse(url).scheme == 'https' and not urlparse(url).username,
             'Anonymous HTTPS URL required')
        base = {'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'}
        base.update(headers or {})
        need(not any(key.lower() in ('authorization', 'cookie') for key in base),
             'Authentication headers are forbidden')
        for attempt in range(3):
            try:
                return self.opener.open(Request(url, headers=base, method='GET'), timeout=self.timeout)
            except HTTPError as error:
                retry = error.code in (429, 500, 502, 503, 504)
                if not retry or attempt == 2:
                    raise
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(attempt + 1)

    def json(self, path):
        root = '/repos/' + GITHUB_REPO
        need(path == root or path.startswith(root + '/'), 'Unexpected API path')
        with self.open(API + path, headers={'Accept': 'application/vnd.github+json',
                                          'X-GitHub-Api-Version': '2026-03-10'}) as response:
            need(response.status == 200, 'GitHub API did not return 200')
            body = response.read(16 * 1024**2 + 1)
        need(len(body) <= 16 * 1024**2, 'Oversized GitHub JSON response')
        return json.loads(body)


def tag_commit(http):
    ref = http.json('/repos/' + GITHUB_REPO + '/git/ref/tags/' + quote(TAG, safe=''))
    need(ref['ref'] == 'refs/tags/' + TAG, 'GitHub tag reference differs')
    obj = ref['object']
    seen = set()
    while obj['type'] == 'tag':
        need(COMMIT.fullmatch(obj['sha']) and obj['sha'] not in seen and len(seen) < 8,
             'Invalid or recursive annotated tag')
        seen.add(obj['sha'])
        annotated = http.json('/repos/' + GITHUB_REPO + '/git/tags/' + obj['sha'])
        need(annotated['sha'] == obj['sha'], 'Annotated tag SHA differs')
        obj = annotated['object']
    need(obj['type'] == 'commit' and COMMIT.fullmatch(obj['sha']), 'Tag does not resolve to a commit')
    return obj['sha']


def inventory(http, release_id, expected):
    need(type(release_id) is int and release_id > 0, 'Invalid release id')
    rows = []
    for page in range(1, 12):
        batch = http.json('/repos/' + GITHUB_REPO + '/releases/' + str(release_id)
                          + '/assets?per_page=100&page=' + str(page))
        need(isinstance(batch, list), 'GitHub asset page must be a list')
        rows.extend(batch)
        if len(batch) < 100:
            break
    else:
        raise ValueError('Asset pagination exceeded the release limit')
    assets = {}
    ids = set()
    for item in rows:
        name = item['name']
        need(name in expected and name not in assets and item['state'] == 'uploaded'
             and type(item['id']) is int and item['id'] > 0 and item['id'] not in ids
             and type(item['size']) is int and item['size'] == expected[name]['bytes'],
             'Unexpected, duplicate, incomplete, or wrong-sized asset: ' + str(name))
        url = 'https://github.com/' + GITHUB_REPO + '/releases/download/' + quote(TAG, safe='') + '/' + quote(name, safe='')
        need(item['browser_download_url'] == url, 'Download URL is not the exact public release asset')
        digest = item.get('digest')
        if digest is not None:
            need(isinstance(digest, str) and digest == 'sha256:' + expected[name]['sha256'],
                 'GitHub API SHA-256 differs: ' + name)
        ids.add(item['id'])
        assets[name] = {'id': item['id'], 'name': name, 'size': item['size'], 'digest': digest,
                        'state': item['state'], 'updated_at': item['updated_at'], 'url': url}
    need(set(assets) == set(expected), 'Public release assets do not exactly match the built 120 assets')
    return assets


def full_download(http, asset, expected, destination=None):
    digest = hashlib.sha256()
    size = 0
    output = None
    if destination is not None:
        output = Path(destination).open('xb')
    try:
        with http.open(asset['url'], headers={'Accept': 'application/octet-stream'}) as response:
            need(response.status == 200 and response.headers.get('Content-Encoding', 'identity') == 'identity',
                 'Full download must return unencoded HTTP 200')
            length = response.headers.get('Content-Length')
            need(length is None or int(length) == expected['bytes'], 'Full download length differs')
            for chunk in iter(lambda: response.read(8 * 1024**2), b''):
                size += len(chunk)
                need(size <= expected['bytes'], 'Full download exceeds expected asset size')
                digest.update(chunk)
                if output is not None:
                    output.write(chunk)
    finally:
        if output is not None:
            output.close()
    need(size == expected['bytes'] and digest.hexdigest() == expected['sha256'],
         'Anonymous complete download SHA-256 or byte count differs: ' + asset['name'])
    return {'anonymous_access_verified': True, 'independent_full_sha256_verified': True,
            'downloaded_bytes': size, 'hash_source': 'anonymous_complete_download',
            'sha256': digest.hexdigest(), 'bytes': size}


def sample_download(http, asset, expected):
    stop = min(1024, expected['bytes']) - 1
    with http.open(asset['url'], headers={'Accept': 'application/octet-stream',
                                        'Range': 'bytes=0-' + str(stop)}) as response:
        status = response.status
        need(status in (200, 206) and response.headers.get('Content-Encoding', 'identity') == 'identity',
             'Weight range request did not return unencoded HTTP 200/206')
        need('text/html' not in response.headers.get('Content-Type', '').lower(),
             'Weight request returned an HTML page')
        content_range = response.headers.get('Content-Range')
        if status == 206:
            need(content_range == 'bytes 0-' + str(stop) + '/' + str(expected['bytes']),
                 'Weight Content-Range differs')
            length = response.headers.get('Content-Length')
            need(length is None or int(length) == stop + 1, 'Weight range byte count differs')
        else:
            length = response.headers.get('Content-Length')
            need(length is None or int(length) == expected['bytes'], 'Weight response size differs')
        # A server may ignore Range. Close after a bounded prefix in that case;
        # do not label this as a full file hash or count unread transport bytes.
        prefix = response.read(stop + 1)
    need(len(prefix) == stop + 1 and len(prefix) >= 9, 'Empty or truncated weight prefix')
    header_length = int.from_bytes(prefix[:8], 'little')
    need(2 <= header_length <= 16 * 1024**2 and header_length + 8 < expected['bytes']
         and prefix[8:9] == b'{', 'Public asset prefix is not a safetensors container')
    return {'anonymous_access_verified': True, 'independent_full_sha256_verified': False,
            'hash_source': 'github_api_digest', 'http_status': status,
            'content_range': content_range, 'downloaded_bytes': len(prefix),
            'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
            'range_ignored': status == 200}


def check_runtime_archive(archive, destination, built, expected):
    """Safely inspect/extract only the small runtime archive and bind its manifest."""
    destination = Path(destination)
    need(not destination.exists(), 'Fresh runtime inspection directory required')
    destination.mkdir()
    actual = {}
    total = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = member.name
            path = relative(name)
            need(path.parts[0] == NAME and len(path.parts) > 1 and member.isfile()
                 and name not in actual and 0 <= member.size <= MAX_RUNTIME_MEMBER,
                 'Unexpected runtime tar member: ' + name)
            local = Path(*path.parts[1:])
            need(local.suffix != '.safetensors', 'Runtime archive unexpectedly contains a weight shard')
            total += member.size
            need(total <= MAX_RUNTIME_UNPACKED, 'Runtime archive unpacked size exceeds inspection limit')
            target = destination / local
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            source = tar.extractfile(member)
            need(source is not None, 'Missing regular runtime member stream')
            with source, target.open('xb') as output:
                for chunk in iter(lambda: source.read(8 * 1024**2), b''):
                    count += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
            need(count == member.size, 'Truncated runtime member')
            actual[str(local)] = {'sha256': digest.hexdigest(), 'bytes': count}
    need(sha_file(destination / 'manifest.json') == built['bundle_manifest_sha256'],
         'Downloaded runtime manifest does not match the frozen build')
    manifest = read(destination / 'manifest.json')
    need(manifest['format'] == BUILD_FORMAT and manifest['complete'] is True
         and manifest['github_repo'] == GITHUB_REPO and manifest['hugging_face_repo'] == HF_REPO
         and manifest['release_tag'] == TAG and manifest['candidate_ppl'] == built['candidate_ppl'],
         'Downloaded runtime manifest identity differs')
    weights = read(destination / 'weights/weight_manifest.json')
    shards = {'weights/' + name for name in weights['files']}
    wanted = set(manifest['files']) | {'manifest.json', 'SHA256SUMS'}
    need(set(actual) == wanted - shards, 'Runtime tar has missing or extra members')
    need({name: item for name, item in expected.items() if name != RUNTIME} == weights['files'],
         'Runtime weight metadata does not bind all 119 public weight assets')
    for name, item in manifest['files'].items():
        relative(name)
        if name in shards:
            need(item == expected[Path(name).name], 'Weight asset differs from complete bundle manifest')
        else:
            need(item == actual[name], 'Downloaded runtime member differs: ' + name)
    hashes = {name: item['sha256'] for name, item in manifest['files'].items()}
    hashes['manifest.json'] = built['bundle_manifest_sha256']
    need((destination / 'SHA256SUMS').read_text() == ''.join(
        digest + '  ' + name + '\n' for name, digest in sorted(hashes.items())),
        'Downloaded runtime checksum list differs')
    # Recheck archived full-quality/audit bindings and all 507 decoded weight
    # ledger hashes. This does not download the shard data or rerun GPU quality.
    comp, train = quality(destination)
    need(manifest['adapter_binding'] == train['binding'] and manifest['state_provenance'] == comp['parent']
         and manifest['backend_policy'] == comp['backend_policy'], 'Runtime recipe binding differs')
    return {'complete': True, 'verified': True, 'members': len(actual), 'unpacked_bytes': total,
            'manifest_sha256': built['bundle_manifest_sha256'], 'bound_full_quality_gate_passed': True,
            'quality_recomputed_on_gpu': False, 'weight_shard_data_downloaded_by_archive_check': False}


def verify(args):
    built = read(args.build_receipt)
    expected = expected_assets(built)
    need(COMMIT.fullmatch(args.expected_commit), 'Exact expected 40-character source commit required')
    need(not args.out.exists() and not args.out.is_symlink(), 'Fresh GitHub receipt path required')
    http = AnonymousHTTP(args.timeout)
    repo = http.json('/repos/' + GITHUB_REPO)
    need(repo['full_name'] == GITHUB_REPO and repo['private'] is False
         and repo.get('visibility', 'public') == 'public' and repo['html_url'] == 'https://github.com/' + GITHUB_REPO,
         'Exact public GitHub repository required')
    release_path = '/repos/' + GITHUB_REPO + '/releases/tags/' + quote(TAG, safe='')
    release = http.json(release_path)
    need(release['tag_name'] == TAG and release['draft'] is False and release['prerelease'] is False
         and release['published_at'] is not None
         and release['html_url'] == 'https://github.com/' + GITHUB_REPO + '/releases/tag/' + TAG,
         'Exact public non-draft, non-prerelease release required')
    need(tag_commit(http) == args.expected_commit, 'Public tag points to a different source commit')
    assets = inventory(http, release['id'], expected)
    checks = {}
    if args.download_dir is not None:
        need(not args.download_dir.exists() and not args.download_dir.is_symlink(), 'Fresh download directory required')
        args.download_dir.mkdir(parents=True)
        working = args.download_dir
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix='mamba2-public-release-check-')
        working = Path(cleanup.name)
    try:
        archive = working / RUNTIME
        checks[RUNTIME] = full_download(http, assets[RUNTIME], expected[RUNTIME], archive)
        runtime_check = check_runtime_archive(archive, working / NAME, built, expected)
        for index, name in enumerate(sorted(set(expected) - {RUNTIME}), 1):
            asset = assets[name]
            checks[name] = (sample_download(http, asset, expected[name]) if asset['digest'] is not None
                            else full_download(http, asset, expected[name]))
            checks[name]['github_api_digest_verified'] = asset['digest'] is not None
            print(json.dumps({'asset': name, 'weight_progress': [index, 119],
                              'hash_source': checks[name]['hash_source'],
                              'bytes_read': checks[name]['downloaded_bytes']}), file=sys.stderr, flush=True)
        checks[RUNTIME]['github_api_digest_verified'] = assets[RUNTIME]['digest'] is not None
        final_release = http.json(release_path)
        need(final_release['id'] == release['id'] and final_release['tag_name'] == TAG
             and final_release['draft'] is False and final_release['prerelease'] is False,
             'Public release changed during verification')
        need(inventory(http, release['id'], expected) == assets and tag_commit(http) == args.expected_commit,
             'Public asset metadata or tag changed during verification')
        result = dict(format=FORMAT, complete=True, verified=True, anonymous_access_verified=True,
            repository=GITHUB_REPO, tag=TAG, commit=args.expected_commit,
            release_id=release['id'], release_url=release['html_url'],
            build_receipt_sha256=sha_file(args.build_receipt),
            bundle_manifest_sha256=built['bundle_manifest_sha256'], candidate_ppl=built['candidate_ppl'],
            assets=expected, asset_checks=checks, runtime_archive_check=runtime_check,
            github_api_digest_verified_assets=sum(row['github_api_digest_verified'] for row in checks.values()),
            independently_full_downloaded_and_hashed_assets=sum(row['independent_full_sha256_verified'] for row in checks.values()),
            anonymously_sampled_weight_assets=sum(not row['independent_full_sha256_verified'] for row in checks.values()),
            application_bytes_read=sum(row['downloaded_bytes'] for row in checks.values()),
            checked_at=datetime.now(timezone.utc).isoformat(), credentials_used=False, remote_mutation_performed=False,
            verification_scope='Anonymous public repository, peeled tag commit and exact 120-asset inventory; '
              'runtime archive fully downloaded and SHA-256 checked with its audited recipe bindings; '
              'weight SHA-256 values matched GitHub API digests and anonymous safetensors prefixes read. '
              'Any asset lacking an API digest was instead fully downloaded and SHA-256 checked. '
              'API digest verification and prefix access are not an independent full download hash of those weights; '
              'GPU PPL/MK quality was not rerun.')
        write_fresh(args.out, result)
        return result
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--download-dir', type=Path,
                        help='Fresh directory retaining the downloaded runtime archive and inspected runtime; weights are not retained')
    parser.add_argument('--timeout', type=int, default=30)
    args = parser.parse_args()
    need(1 <= args.timeout <= 60, 'Per-request timeout must be 1..60 seconds')
    try:
        result = verify(args)
    except Exception as error:
        # Failure never produces a receipt consumable by the HF stager.
        if not args.out.exists() and not args.out.is_symlink():
            write_fresh(args.out, {'format': FORMAT, 'complete': False, 'verified': False,
                                  'anonymous_access_verified': False,
                                  'error_type': type(error).__name__, 'error': str(error)[:500],
                                  'credentials_used': False, 'remote_mutation_performed': False})
        raise
    print(json.dumps({key: result[key] for key in ('complete', 'verified', 'repository', 'tag', 'commit',
          'github_api_digest_verified_assets', 'independently_full_downloaded_and_hashed_assets',
          'anonymously_sampled_weight_assets', 'application_bytes_read')}, indent=2))


if __name__ == '__main__':
    main()
