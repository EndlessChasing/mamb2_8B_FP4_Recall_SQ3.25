#!/usr/bin/env python3
"""Export or verify actual packed FP4 G16 weights; no publication or Resurface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall.fp4_checkpoint import export_checkpoint, verify_checkpoint, sha_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export', type=Path, help='Fresh output checkpoint directory')
    mode.add_argument('--verify', type=Path, help='Check packed files/ledger without torch or CUDA')
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--conversion-receipt', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--shard-limit-mib', type=int, default=256)
    args = parser.parse_args()
    if args.export:
        if args.source_dir is None or args.conversion_receipt is None:
            parser.error('--export requires --source-dir and --conversion-receipt')
        manifest = export_checkpoint(args.source_dir, args.conversion_receipt, args.export,
                                    device=args.device, shard_limit_bytes=args.shard_limit_mib * 1024 * 1024,
                                    progress=lambda row: print('[FP4 export] ' + json.dumps(row), flush=True))
        directory = args.export
    else:
        manifest, _ = verify_checkpoint(args.verify)
        directory = args.verify
    print(json.dumps({'verified': True, 'weight_manifest_sha256': sha_file(directory / 'weight_manifest.json'),
                      'logical_weight_payload_bytes': manifest['logical_weight_payload_bytes'],
                      'shard_file_bytes': manifest['shard_file_bytes'],
                      'shards': len(manifest['files']), 'decoded_resident_weight_bytes': manifest['decoded_resident_weight_bytes'],
                      'scope': manifest['scope']}, indent=2))


if __name__ == '__main__':
    main()
