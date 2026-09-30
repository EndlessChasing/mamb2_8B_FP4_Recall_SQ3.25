#!/usr/bin/env python3
"""Verify the complete FP4/SQ3.25 public Hugging Face snapshot anonymously.

Standard library only; GET requests only; no credentials, cookies, netrc, HF
configuration or authenticated proxy are read. All nonweight files are fetched
and independently hashed. The 119 weight files are checked against HF LFS
SHA-256 metadata and sampled through bounded anonymous prefix requests. Those
samples are not an independent SHA-256 hash of the complete large files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

FORMAT = 'MAMBA2_FP4_G16_SQ325_HF_PUBLICATION_CHECK_V1'
STAGE_FORMAT = 'MAMBA2_FP4_G16_SQ325_HF_PAYLOAD_V1'
HF_REPO = 'EndlessChasing/Mamb2_8B_FP4_Recall_SQ3.25'
GH_REPO = 'EndlessChasing/mamb2_8B_FP4_Recall_SQ3.25'
TAG = 'v0.1.0-fp4g16-sq325-resurface'
BASE = 'https://huggingface.co'
USER_AGENT = 'Mamba2-FP4-SQ325-anonymous-HF-verifier/1'
SHA256 = re.compile(r'[0-9a-f]{64}')
COMMIT = re.compile(r'[0-9a-f]{40}')
API_ROOT = '/api/models/' + HF_REPO
MAX_JSON = 16 * 1024**2
MAX_NONWEIGHT = 128 * 1024**2
MAX_FILES = 10000
PREFIX_BYTES = 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(name):
    need(isinstance(name, str) and name and '\\' not in name and '\n' not in name
         and '\r' not in name and '\x00' not in name, 'Invalid publication path')
    path = PurePosixPath(name)
    need(not path.is_absolute() and '..' not in path.parts and str(path) == name,
         'Unsafe publication path: ' + name)
    return path


def write_fresh(path, value):
    path = Path(path)
    need(not path.exists() and not path.is_symlink(), 'Fresh verification receipt required')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def expected_files(manifest_path, checksums_path):
    """Use only two small local files; the 4.6 GB stage need not be copied here."""
    manifest_path, checksums_path = Path(manifest_path), Path(checksums_path)
    need(manifest_path.is_file() and not manifest_path.is_symlink()
         and checksums_path.is_file() and not checksums_path.is_symlink(),
         'Regular local publish manifest and checksum list required')
    manifest = read(manifest_path)
    need(manifest['format'] == STAGE_FORMAT and manifest['complete'] is True
         and manifest['ready_for_upload'] is True and manifest['repository'] == HF_REPO
         and manifest['github_repository'] == GH_REPO and manifest['github_tag'] == TAG
         and COMMIT.fullmatch(manifest['github_commit'])
         and SHA256.fullmatch(manifest['github_receipt_sha256'])
         and SHA256.fullmatch(manifest['bundle_manifest_sha256'])
         and isinstance(manifest['candidate_ppl'], (float, int))
         and math.isfinite(manifest['candidate_ppl']) and 0 < manifest['candidate_ppl'] < 8.
         and manifest['baseline_ppl'] == 8.408282583578627,
         'Exact condition-passing GitHub-first HF staging manifest required')
    expected = {}
    weights = set()
    for name, value in manifest['files'].items():
        path = relative(name)
        need(name not in ('publish_manifest.json', 'SHA256SUMS')
             and SHA256.fullmatch(value['sha256']) and type(value['bytes']) is int
             and value['bytes'] >= 0, 'Invalid staged file metadata: ' + name)
        expected[name] = {'sha256': value['sha256'], 'bytes': value['bytes']}
        if value['source'] == 'verified_github_weight_asset':
            need(path.parent == PurePosixPath('release/weights') and path.suffix == '.safetensors'
                 and value['asset_name'] == path.name and value['bytes'] > 8,
                 'Unexpected declared weight asset: ' + name)
            weights.add(name)
        elif value['source'] == 'verified_github_runtime_archive':
            need(name.startswith('release/') and value['bytes'] <= MAX_NONWEIGHT,
                 'Unexpected runtime file: ' + name)
        else:
            need((name, value['source']) in {('README.md', 'reviewed_hf_card'),
                                            ('LICENSES.md', 'license_notice'),
                                            ('.gitattributes', 'lfs_attributes')},
                 'Unexpected root file: ' + name)
        if name.startswith('release/'):
            need(value['github_repository'] == GH_REPO and value['github_tag'] == TAG
                 and value['github_commit'] == manifest['github_commit'],
                 'File GitHub provenance differs: ' + name)
    need(len(weights) == 119 and sum(expected[name]['bytes'] for name in weights) == 4638539864,
         'Exactly 119 measured weight containers and their complete byte count required')
    need(len(expected) == manifest['file_count_excluding_root_manifest_checksums']
         and sum(item['bytes'] for item in expected.values()) == manifest['file_bytes_excluding_root_manifest_checksums']
         and 0 < len(expected) < MAX_FILES, 'Staged file count or byte total differs')
    for name in ('README.md', 'LICENSES.md', '.gitattributes', 'release/manifest.json',
                 'release/SHA256SUMS', 'release/weights/weight_manifest.json',
                 'release/adapter_fp16.pt', 'release/state_config.pt',
                 'release/evidence/comparison.json', 'release/evidence/audit.json'):
        need(name in expected, 'Required publication file absent: ' + name)
    need(expected['release/manifest.json']['sha256'] == manifest['bundle_manifest_sha256'],
         'Outer bundle manifest identity differs')
    expected['publish_manifest.json'] = {'sha256': sha_file(manifest_path), 'bytes': manifest_path.stat().st_size}
    expected_checksums = ''.join(f"{item['sha256']}  {name}\n" for name, item in sorted(expected.items())).encode()
    need(checksums_path.read_bytes() == expected_checksums, 'Local HF checksum list does not exactly bind staged files')
    expected['SHA256SUMS'] = {'sha256': sha_file(checksums_path), 'bytes': checksums_path.stat().st_size}
    return manifest, expected, weights


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        need(parsed.scheme == 'https' and parsed.hostname and parsed.username is None
             and parsed.password is None, 'Anonymous redirects must remain HTTPS without URL credentials')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        need(redirected is None or (redirected.get_method() == 'GET'
             and not any(key.lower() in ('authorization', 'cookie', 'proxy-authorization')
                         for key, _ in redirected.header_items())), 'Authentication or non-GET redirect forbidden')
        return redirected


class AnonymousHTTP:
    def __init__(self, timeout=30., ca_file=None):
        context = ssl.create_default_context(cafile=None if ca_file is None else str(ca_file))
        self.opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context), HTTPSRedirects())
        self.timeout = timeout

    def open(self, url, extra_headers=None):
        parsed = urlparse(url)
        need(parsed.scheme == 'https' and parsed.hostname and parsed.username is None
             and parsed.password is None, 'Anonymous HTTPS URL required')
        headers = {'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'}
        headers.update(extra_headers or {})
        need(not any(key.lower() in ('authorization', 'cookie', 'proxy-authorization') for key in headers),
             'Authentication headers forbidden')
        for attempt in range(3):
            try:
                return self.opener.open(Request(url, headers=headers, method='GET'), timeout=self.timeout)
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                error.close()
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(attempt + 1)

    def json(self, url):
        parsed = urlparse(url)
        need(parsed.scheme == 'https' and parsed.netloc == 'huggingface.co'
             and (parsed.path == API_ROOT or parsed.path.startswith(API_ROOT + '/')),
             'Unexpected HF model API URL')
        with self.open(url, {'Accept': 'application/json'}) as response:
            final = urlparse(response.geturl())
            need(response.status == 200 and final.netloc == parsed.netloc and final.path == parsed.path
                 and response.headers.get('Content-Encoding', 'identity') == 'identity',
                 'HF API must return unencoded HTTP 200 at the requested model endpoint')
            body = response.read(MAX_JSON + 1)
            link = response.headers.get('Link')
        need(len(body) <= MAX_JSON, 'HF API JSON exceeds inspection limit')
        return json.loads(body), link


def model_info(http, revision, commit):
    url = BASE + API_ROOT + '/revision/' + quote(revision, safe='') + '?blobs=true'
    info, _ = http.json(url)
    need(isinstance(info, dict) and info['id'] == HF_REPO and info['private'] is False
         and info['gated'] is False and info['sha'] == commit,
         'Model must be public, ungated, and resolve to the expected commit: ' + revision)
    need(isinstance(info['siblings'], list), 'HF model info must include complete blob metadata')
    result = {}
    for row in info['siblings']:
        name = row['rfilename']
        relative(name)
        need(name not in result and COMMIT.fullmatch(row['blobId'])
             and type(row['size']) is int and row['size'] >= 0, 'Invalid/duplicate HF blob metadata: ' + name)
        lfs = row.get('lfs')
        if lfs is not None:
            need(isinstance(lfs, dict) and SHA256.fullmatch(lfs['sha256'])
                 and lfs['size'] == row['size'] and type(lfs['pointerSize']) is int
                 and lfs['pointerSize'] > 0, 'Invalid LFS blob metadata: ' + name)
        result[name] = {'git_blob_sha1': row['blobId'], 'bytes': row['size'],
                        'lfs_sha256': None if lfs is None else lfs['sha256'],
                        'lfs_pointer_bytes': None if lfs is None else lfs['pointerSize']}
    need(0 < len(result) <= MAX_FILES, 'HF model file inventory exceeds inspection limit')
    return result


def next_tree_page(link, tree_path):
    if not link:
        return None
    matches = re.findall(r'<([^<>]+)>\s*;\s*rel="([^"]+)"', link)
    need(matches, 'Unsupported HF pagination Link header')
    next_links = [url for url, relation in matches if 'next' in relation.split()]
    need(len(next_links) <= 1, 'Multiple HF next-page links')
    if not next_links:
        return None
    url = next_links[0]
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    need(parsed.scheme == 'https' and parsed.netloc == 'huggingface.co' and parsed.path == tree_path
         and set(query) == {'recursive', 'expand', 'limit', 'cursor'}
         and query['recursive'] == ['true'] and query['expand'] == ['false']
         and query['limit'] == ['100'] and len(query['cursor']) == 1 and query['cursor'][0],
         'HF pagination left the pinned anonymous tree endpoint')
    return url


def tree_inventory(http, commit):
    tree_path = API_ROOT + '/tree/' + commit
    url = BASE + tree_path + '?recursive=true&expand=false&limit=100'
    seen_urls, seen_paths, files = set(), set(), {}
    for _ in range(101):
        need(url not in seen_urls, 'HF tree pagination cycle')
        seen_urls.add(url)
        rows, link = http.json(url)
        need(isinstance(rows, list), 'HF tree page must be a JSON list')
        for row in rows:
            name = row['path']
            relative(name)
            need(name not in seen_paths and COMMIT.fullmatch(row['oid']), 'Invalid/duplicate HF tree object: ' + name)
            seen_paths.add(name)
            need(row['type'] in ('file', 'directory'), 'Unsupported HF tree entry type: ' + name)
            if row['type'] == 'directory':
                continue
            need(type(row['size']) is int and row['size'] >= 0, 'Invalid HF file size: ' + name)
            lfs = row.get('lfs')
            if lfs is not None:
                need(isinstance(lfs, dict) and SHA256.fullmatch(lfs['oid'])
                     and lfs['size'] == row['size'] and type(lfs['pointerSize']) is int
                     and lfs['pointerSize'] > 0, 'Invalid HF tree LFS metadata: ' + name)
            files[name] = {'git_blob_sha1': row['oid'], 'bytes': row['size'],
                           'lfs_sha256': None if lfs is None else lfs['oid'],
                           'lfs_pointer_bytes': None if lfs is None else lfs['pointerSize']}
        need(len(seen_paths) <= MAX_FILES, 'HF recursive tree exceeds inspection limit')
        url = next_tree_page(link, tree_path)
        if url is None:
            return files, len(seen_urls)
    raise ValueError('HF tree pagination exceeded inspection limit')


def match_metadata(blobs, tree, expected, weights):
    need(blobs == tree and set(blobs) == set(expected),
         'HF blob metadata, paginated tree, and expected stage inventory must exactly match')
    for name, item in expected.items():
        row = blobs[name]
        need(row['bytes'] == item['bytes'], 'HF file byte count differs: ' + name)
        if row['lfs_sha256'] is not None:
            need(row['lfs_sha256'] == item['sha256'], 'HF LFS content SHA-256 differs: ' + name)
        if name in weights:
            need(row['lfs_sha256'] == item['sha256'], 'Every declared weight must have matching LFS SHA-256 metadata: ' + name)


def resolve_url(commit, name):
    relative(name)
    return BASE + '/' + HF_REPO + '/resolve/' + commit + '/' + quote(name, safe='/')


def full_download(http, commit, name, expected, metadata):
    need(expected['bytes'] <= MAX_NONWEIGHT, 'Nonweight download exceeds inspection limit: ' + name)
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(b'blob ' + str(expected['bytes']).encode() + b'\0')
    count = 0
    retained = bytearray() if name in ('README.md', 'publish_manifest.json', 'SHA256SUMS') else None
    with http.open(resolve_url(commit, name), {'Accept': 'application/octet-stream'}) as response:
        need(response.status == 200 and response.headers.get('Content-Encoding', 'identity') == 'identity',
             'Full anonymous file download must return unencoded HTTP 200: ' + name)
        length = response.headers.get('Content-Length')
        need(length is None or int(length) == expected['bytes'], 'Full file Content-Length differs: ' + name)
        resolved_commit = response.headers.get('X-Repo-Commit')
        need(resolved_commit is None or resolved_commit == commit, 'Downloaded file resolved to a different commit: ' + name)
        while True:
            chunk = response.read(min(8 * 1024**2, expected['bytes'] - count + 1))
            if not chunk:
                break
            count += len(chunk)
            need(count <= expected['bytes'], 'Full download exceeds expected file size: ' + name)
            sha256.update(chunk)
            git_blob.update(chunk)
            if retained is not None:
                retained.extend(chunk)
    need(count == expected['bytes'] and sha256.hexdigest() == expected['sha256'],
         'Independent full anonymous SHA-256 or byte count differs: ' + name)
    if metadata['lfs_sha256'] is None:
        need(git_blob.hexdigest() == metadata['git_blob_sha1'], 'Independent Git blob SHA-1 differs: ' + name)
    if name == 'README.md':
        need(bytes(retained).startswith(b'---\n') and HF_REPO.encode() in retained
             and b'DRAFT' not in retained and re.search(rb'\{\{[^{}\n]{1,120}\}\}', retained) is None,
             'Public model card contains draft markers or wrong identity')
    return {'anonymous_access_verified': True, 'independent_full_sha256_verified': True,
            'hash_source': 'anonymous_complete_download', 'downloaded_bytes': count,
            'sha256': sha256.hexdigest(), 'bytes': count,
            'lfs_metadata_sha256_verified': metadata['lfs_sha256'] is not None,
            'git_blob_sha1_verified': metadata['lfs_sha256'] is None}, retained


def weight_prefix(http, commit, name, expected, metadata):
    prefix_bytes = min(PREFIX_BYTES, expected['bytes'])
    stop = prefix_bytes - 1
    with http.open(resolve_url(commit, name), {'Accept': 'application/octet-stream',
                                              'Range': 'bytes=0-' + str(stop)}) as response:
        status = response.status
        need(status in (200, 206) and response.headers.get('Content-Encoding', 'identity') == 'identity'
             and 'text/html' not in response.headers.get('Content-Type', '').lower(),
             'Anonymous weight prefix did not return binary HTTP 200/206: ' + name)
        content_range = response.headers.get('Content-Range')
        length = response.headers.get('Content-Length')
        if status == 206:
            need(content_range == f'bytes 0-{stop}/{expected["bytes"]}'
                 and (length is None or int(length) == prefix_bytes), 'Weight Content-Range differs: ' + name)
        else:
            need(length is None or int(length) == expected['bytes'], 'Weight response byte count differs: ' + name)
        prefix = response.read(prefix_bytes)
        # Close immediately when Range is ignored. Read bytes are bounded; the
        # receipt does not claim to measure unread network transport buffering.
    need(len(prefix) == prefix_bytes and prefix_bytes >= 9,
         'Anonymous safetensors prefix is empty/truncated: ' + name)
    header_size = int.from_bytes(prefix[:8], 'little')
    need(2 <= header_size <= MAX_JSON and header_size + 8 < expected['bytes']
         and prefix[8:9] == b'{', 'Weight prefix is not a safetensors container: ' + name)
    need(metadata['lfs_sha256'] == expected['sha256'], 'Weight LFS SHA-256 differs: ' + name)
    return {'anonymous_access_verified': True, 'independent_full_sha256_verified': False,
            'hash_source': 'hf_lfs_api_sha256_metadata', 'lfs_metadata_sha256_verified': True,
            'sha256': expected['sha256'], 'bytes': expected['bytes'],
            'http_status': status, 'content_range': content_range,
            'range_ignored': status == 200, 'downloaded_bytes': len(prefix),
            'prefix_sha256': hashlib.sha256(prefix).hexdigest()}


def verify(args):
    need(COMMIT.fullmatch(args.expected_commit) and args.tag == TAG,
         'Exact 40-character expected HF commit and fixed release tag required')
    need(1 <= args.workers <= 8 and 0 < args.timeout <= 60., 'Invalid HTTP worker/timeout limit')
    need(not args.out.exists(), 'Fresh publication verification output required')
    manifest, expected, weights = expected_files(args.publish_manifest, args.checksums)
    http = AnonymousHTTP(args.timeout, args.ca_file)
    initial = model_info(http, args.expected_commit, args.expected_commit)
    need(model_info(http, args.tag, args.expected_commit) == initial
         and model_info(http, 'main', args.expected_commit) == initial,
         'Public commit, release tag, and main file metadata differ')
    tree, pages = tree_inventory(http, args.expected_commit)
    match_metadata(initial, tree, expected, weights)
    checks = {}
    retained = {}

    def task(name):
        # Each concurrent request gets its own stateless anonymous opener.
        client = AnonymousHTTP(args.timeout, args.ca_file)
        if name in weights:
            return name, weight_prefix(client, args.expected_commit, name, expected[name], initial[name]), None
        result, body = full_download(client, args.expected_commit, name, expected[name], initial[name])
        return name, result, body

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(task, name) for name in sorted(expected)]
        for future in as_completed(pending):
            name, result, body = future.result()
            checks[name] = result
            if body is not None:
                retained[name] = bytes(body)
            if len(checks) == 1 or len(checks) % 32 == 0 or len(checks) == len(expected):
                print(f'[anonymous HF verification] {len(checks)}/{len(expected)} files', flush=True)
    need(json.loads(retained['publish_manifest.json']) == manifest
         and retained['publish_manifest.json'] == args.publish_manifest.read_bytes()
         and retained['SHA256SUMS'] == args.checksums.read_bytes(),
         'Public publish manifest/checksum bytes differ from reviewed local stage')
    # Pinned commit data are immutable. Recheck mutable references and public
    # visibility once after inspection, rather than repeatedly polling them.
    need(model_info(http, args.tag, args.expected_commit) == initial
         and model_info(http, 'main', args.expected_commit) == initial,
         'Public tag/main changed during verification')
    full_names = set(expected) - weights
    need(all(checks[name]['independent_full_sha256_verified'] for name in full_names)
         and all(not checks[name]['independent_full_sha256_verified'] for name in weights),
         'Full-download and metadata/prefix verification scopes differ')
    receipt = {
        'format': FORMAT, 'complete': True, 'verified': True, 'anonymous_access_verified': True,
        'repository': HF_REPO, 'commit': args.expected_commit, 'tag': args.tag,
        'public': True, 'gated': False, 'main_and_tag_expected_commit_verified': True,
        'publication_performed': False, 'new_quality_evaluation_performed': False,
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha_file(Path(__file__)),
        'publish_manifest_sha256': sha_file(args.publish_manifest),
        'checksums_sha256': sha_file(args.checksums),
        'bundle_manifest_sha256': manifest['bundle_manifest_sha256'],
        'github_repository': GH_REPO, 'github_tag': TAG, 'github_commit': manifest['github_commit'],
        'github_receipt_sha256': manifest['github_receipt_sha256'],
        'candidate_ppl': manifest['candidate_ppl'], 'baseline_ppl': manifest['baseline_ppl'],
        'exact_file_inventory_verified': True, 'tree_pages': pages,
        'file_count': len(expected), 'total_file_bytes': sum(item['bytes'] for item in expected.values()),
        'independently_full_hashed_file_count': len(full_names),
        'independently_full_hashed_file_bytes': sum(expected[name]['bytes'] for name in full_names),
        'lfs_metadata_sha256_verified_file_count': sum(row['lfs_sha256'] is not None for row in initial.values()),
        'weight_file_count': len(weights), 'weight_container_bytes': 4638539864,
        'independently_full_hashed_weight_count': 0, 'all_weights_downloaded_in_full': False,
        'weight_prefix_bytes_read': sum(checks[name]['downloaded_bytes'] for name in weights),
        'verification_scope': 'Anonymous GET only. Exact public ungated model, expected tag/main commit, blob/tree inventory and bytes. All nonweight files independently downloaded and SHA-256 hashed; 119 weights matched to HF LFS API SHA-256 metadata and anonymously read through bounded prefixes. Prefixes are not independent full weight hashes.',
        'files': {name: checks[name] for name in sorted(checks)},
    }
    write_fresh(args.out, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-manifest', type=Path, required=True)
    parser.add_argument('--checksums', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--tag', default=TAG)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=30.)
    parser.add_argument('--ca-file', type=Path, help='Explicit trusted CA bundle when the system Python trust store is unavailable')
    args = parser.parse_args()
    try:
        receipt = verify(args)
    except HTTPError as error:
        # Do not print a CDN redirect URL, which can include public presigned
        # download parameters. No account credential is loaded by this helper.
        print('Anonymous HF verification failed: HTTP ' + str(error.code), file=sys.stderr)
        raise SystemExit(1) from None
    except (URLError, TimeoutError, ssl.SSLError) as error:
        print('Anonymous HF verification failed: network/TLS ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
    except (ValueError, KeyError, TypeError) as error:
        print('Anonymous HF verification failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1) from None
    summary = {key: receipt[key] for key in ('complete', 'verified', 'repository', 'commit', 'tag',
               'candidate_ppl', 'file_count', 'independently_full_hashed_file_count',
               'weight_file_count', 'independently_full_hashed_weight_count', 'verification_scope')}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
