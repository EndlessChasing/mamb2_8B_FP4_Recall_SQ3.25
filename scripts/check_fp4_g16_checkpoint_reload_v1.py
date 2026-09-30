#!/usr/bin/env python3
"""Reload the complete real FP4 checkpoint with no original NVIDIA weights.

Checks all 507 decoded tensor hashes, exact resident weight payload and native
geometry. PPL/MK and the final state/Resurface configuration are separate gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall.fp4_checkpoint import load_packed_model, sha_file, need


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists() and not args.out.is_symlink(), 'Fresh reload receipt required')
    records = []
    def progress(row):
        records.append(row)
        print('[FP4 reload] ' + json.dumps(row), flush=True)
    model = load_packed_model(args.checkpoint_dir, device=args.device, progress=progress)
    state = model.state_dict()
    resident = sum(value.untyped_storage().nbytes() for value in state.values())
    need(len(records) == 507 and len(state) == 507 and resident == 16_473_999_360,
         'Incomplete tensor coverage or wrong resident storage')
    report = {'complete': True, 'passed': True, 'source_checkpoint_required': False,
              'weight_manifest_sha256': sha_file(args.checkpoint_dir / 'weight_manifest.json'),
              'conversion_receipt_sha256': sha_file(args.checkpoint_dir / 'conversion_receipt.json'),
              'decoded_tensor_count': len(records), 'decoded_hash_records': records,
              'resident_fp16_weight_bytes': resident, 'packed_resident_kernel': False,
              'scope': 'All decoded tensors match the fixed 8B quality ledger; PPL/MK/state/adapter checks are separate'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'decoded_hash_records'}, indent=2))


if __name__ == '__main__':
    main()
