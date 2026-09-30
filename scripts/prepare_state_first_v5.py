#!/usr/bin/env python3
"""Derive the four frozen v5 state tables on CPU; no adapter or heldout data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mamba2_recall import runtime, resurface_data as data
from prepare_quant_first import TRAIN_SHA, PROSE_MANIFEST_SHA, PROTOCOL_SHA as NUMERIC_PROTOCOL_SHA
from prepare_quant_first import write_new_json

PROTOCOL_SHA = 'ce700910f27230e062fe62e89dd6bddee05bcbf7bb7972c59fdba24cb3db9f5d'
V4_STATISTICS_SHA = '8509bb266d40875608f3e0b3be22407fd6ea1b8aacbc79ce04e958726516b0a4'
ORIGINAL_CALIBRATION_SHA = 'c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023'
TRAIN_MANIFEST_SHA = '451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0'
KINDS = ('magnitude', 'full_readout', 'preserve_int8', 'preserve_retained80')
CACHE_BYTES = 28499968
CANDIDATES_FORMAT = 'MAMBA2_STATE_FIRST_CANDIDATES_V1'
CALIBRATION_FORMAT = 'MAMBA2_STATE_FIRST_CALIBRATION_V1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def base_binding():
    return dict(protocol_sha256=PROTOCOL_SHA, numeric_protocol_sha256=NUMERIC_PROTOCOL_SHA,
        source_sha256=runtime.SOURCE_CHECKPOINT_SHA256, tokenizer_sha256=runtime.TOKENIZER_SHA256,
        train_file_sha256=TRAIN_SHA, prose_manifest_sha256=PROSE_MANIFEST_SHA,
        train_manifest_sha256=TRAIN_MANIFEST_SHA, v4_statistics_sha256=V4_STATISTICS_SHA,
        original_calibration_sha256=ORIGINAL_CALIBRATION_SHA, adapter=None, adapter_sha256=None,
        heldout_used=False, calibration_tokens=4096)


def check_protocol():
    need(data.sha_file(ROOT/'docs/STATE_FIRST_V5_PROTOCOL.md') == PROTOCOL_SHA, 'v5 protocol changed')
    need(data.sha_file(ROOT/'docs/QUANT_FIRST_PROTOCOL.md') == NUMERIC_PROTOCOL_SHA, 'Numeric protocol changed')
    need(data.sha_file(ROOT/'docs/prose_train_manifest.json') == PROSE_MANIFEST_SHA, 'TRAIN prose manifest changed')


def code_hashes():
    paths = ['scripts/prepare_state_first_v5.py', 'scripts/prepare_state_repair.py',
        'scripts/prepare_quant_first.py', 'scripts/run_state_repair.py',
        'mamba2_recall/runtime.py', 'mamba2_recall/resurface_data.py',
        'docs/STATE_FIRST_V5_PROTOCOL.md', 'docs/STATE_REPAIR_PROTOCOL.md']
    return {p:data.sha_file(ROOT/p) for p in paths}


def check_table(table):
    need(isinstance(table, torch.Tensor) and table.device.type == 'cpu'
         and table.dtype == torch.uint8 and tuple(table.shape) == (56, 8, 128), 'Invalid CPU table')
    need(torch.equal(table.sort(-1).values, torch.arange(128, dtype=torch.uint8).expand_as(table)),
         'Each table row must contain exactly coordinates0..127')


def derive_tables(old, readout):
    """Filter the stable global score ordering while preserving specified old sets."""
    check_table(old); check_table(readout)
    old8 = torch.zeros_like(old, dtype=torch.bool).scatter_(-1, old[..., :16].long(), True)
    old80 = torch.zeros_like(old, dtype=torch.bool).scatter_(-1, old[..., :80].long(), True)
    ranked8 = old8.gather(-1, readout.long())
    ranked80 = old80.gather(-1, readout.long())
    rest112 = readout[~ranked8].reshape(56, 8, 112)
    retained80 = readout[ranked80].reshape(56, 8, 80)
    result = dict(magnitude=old.clone().contiguous(), full_readout=readout.clone().contiguous(),
        preserve_int8=torch.cat([old[..., :16], rest112], -1).contiguous(),
        preserve_retained80=torch.cat([retained80, old[..., 80:]], -1).contiguous())
    for table in result.values():
        check_table(table)
    return result


def choose_candidate(results):
    def valid(row):
        return (row.get('complete') is True and 'error' not in row
            and math.isfinite(row['ppl']['ppl']) and row['cache']['total_bytes'] == CACHE_BYTES
            and row.get('persistent_float_finite_checks_passed') is True
            and row.get('repeated_reset_probe', {}).get('hidden_and_cache_exact') is True
            and row.get('frozen_source', {}).get('identity_version_gradients_unchanged') is True
            and row.get('backend_policy_check', {}).get('singleton_config_unchanged') is True
            and row.get('adapter_loaded') is False and row.get('adapter_sha256') is None)
    baseline = results['magnitude']
    need(valid(baseline), 'Magnitude baseline must be valid')
    guard = max(1, math.ceil(.5 * baseline['mk']['summary']['normal']['correct']))
    valid_names = [name for name in KINDS if valid(results[name])]
    admissible = [name for name in valid_names if name == 'magnitude'
                  or results[name]['mk']['summary']['normal']['correct'] >= guard]
    selected = min(admissible, key=lambda name:(results[name]['ppl']['ppl'],
        -results[name]['mk']['summary']['normal']['correct'], KINDS.index(name)))
    return dict(selected_kind=selected, valid=valid_names, admissible=admissible, minimum_mk_correct=guard,
        baseline_mk_correct=baseline['mk']['summary']['normal']['correct'],
        stopped=selected == 'magnitude', adapter_used=False, heldout_used=False,
        rule='Fixed v5: finite complete exact-budget; new MK>=max(1,ceil(.5*baseline)); lowest PPL, highest MK, candidate order; magnitude fallback stops')


def write_payload(path, payload):
    need(not path.exists() and not path.is_symlink(), 'Fresh payload path required')
    pending = path.with_name(path.name+'.pending')
    with pending.open('xb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def load_candidates(path):
    check_protocol()
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    receipt = read_json(path.with_suffix('.json'))
    need(payload.get('format') == receipt.get('format') == CANDIDATES_FORMAT, 'Candidate format differs')
    for key, value in base_binding().items():
        need(payload.get(key) == receipt.get(key) == value, 'Candidate binding differs: '+key)
    need(receipt.get('complete') is True and receipt.get('fresh_source_no_adapter') is True
         and receipt.get('sha256') == data.sha_file(path) and receipt.get('bytes') == path.stat().st_size,
         'Candidate payload/receipt identity differs')
    need(tuple(payload['tables']) == KINDS and tuple(receipt['table_sha256']) == KINDS, 'Candidate order differs')
    for relative, digest in receipt['code_sha256'].items():
        need(data.sha_file(ROOT/relative) == digest, 'Preparation code changed: '+relative)
    derived = derive_tables(payload['tables']['magnitude'], payload['tables']['full_readout'])
    for name in KINDS:
        table = payload['tables'][name]
        need(torch.equal(derived[name], table) and tensor_sha(table) == receipt['table_sha256'][name],
             'Candidate table derivation/hash differs: '+name)
    need(payload['train_token_hashes'] == receipt['train_token_hashes'] and len(payload['train_token_hashes']) == 8,
         'Calibration TRAIN selection differs')
    return payload, receipt


def load_selected_calibration(path, candidates_path):
    path = Path(path)
    candidates, candidates_receipt = load_candidates(candidates_path)
    selected = torch.load(path, map_location='cpu', weights_only=True)
    receipt = read_json(path.with_suffix('.json'))
    selection_path = path.parent/'screen_comparison.json'
    selection = read_json(selection_path)
    need(selected.get('format') == receipt.get('format') == CALIBRATION_FORMAT, 'Selected calibration format differs')
    name = selected.get('selected_kind')
    need(name in KINDS and name != 'magnitude', 'Only a non-magnitude winner may train')
    expected = dict(base_binding(), selected_kind=name, candidates_sha256=data.sha_file(candidates_path),
        selection_report_sha256=data.sha_file(selection_path), table_sha256=candidates_receipt['table_sha256'][name],
        train_token_hashes=candidates['train_token_hashes'])
    for key, value in expected.items():
        need(selected.get(key) == receipt.get(key) == value, 'Selected calibration binding differs: '+key)
    need(receipt.get('complete') is True and receipt.get('fresh_source_no_adapter') is True
         and receipt['sha256'] == data.sha_file(path) and receipt['bytes'] == path.stat().st_size,
         'Selected calibration payload/receipt differs')
    need(selection.get('format') == 'MAMBA2_STATE_FIRST_COMPARISON_V1' and selection.get('stage') == 'screen'
         and selection.get('complete') is True and selection.get('protocol_sha256') == PROTOCOL_SHA
         and selection.get('candidates_sha256') == expected['candidates_sha256']
         and selection.get('adapter_sha256') is None, 'No-adapter selection receipt differs')
    results = {}
    for arm in (*KINDS, 'restored_magnitude'):
        arm_path = path.parent/('screen_'+arm+'.json')
        need(data.sha_file(arm_path) == selection['report_sha256'][arm], 'Selection raw arm hash differs: '+arm)
        row = read_json(arm_path)
        need(row.get('stage') == 'screen' and row.get('adapter_sha256') is None
             and row.get('adapter_loaded') is False and row.get('dataset', {}).get('split') == 'train',
             'Selection must use unadapted TRAIN arms')
        results[arm] = row
    need(selection['selection'] == choose_candidate(results) and selection['selection']['selected_kind'] == name,
         'Frozen TRAIN selection arithmetic differs')
    from evaluate_quant_first import check_restoration
    replay = check_restoration(results['magnitude'], results['restored_magnitude'])
    need(all(selection['restoration'].get(k) == v for k, v in replay.items() if k != 'scope'),
         'Selected calibration lacks exact magnitude restoration')
    need(selection['selected_table_sha256'] == expected['table_sha256'], 'Selected table not frozen in report')
    check_table(selected['permutations'])
    need(torch.equal(selected['permutations'], candidates['tables'][name]), 'Selected coordinates differ')
    for relative, digest in receipt['code_sha256'].items():
        need(data.sha_file(ROOT/relative) == digest, 'Selection implementation changed: '+relative)
    return selected, receipt, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-calibration', type=Path, required=True)
    parser.add_argument('--v4-calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Set CUDA_VISIBLE_DEVICES= for CPU-only preparation')
    check_protocol()
    need(not args.out.exists(), 'Fresh candidate directory required')
    need(data.sha_file(args.original_calibration) == ORIGINAL_CALIBRATION_SHA, 'Pinned magnitude calibration differs')
    need(data.sha_file(args.v4_calibration) == V4_STATISTICS_SHA, 'Pinned S16 TRAIN statistics differ')
    original = torch.load(args.original_calibration, map_location='cpu', weights_only=True)
    v4 = torch.load(args.v4_calibration, map_location='cpu', weights_only=True)
    v4_receipt = read_json(args.v4_calibration.with_suffix('.json'))
    from run_state_repair import validate_repair
    validate_repair(v4, v4_receipt, args.v4_calibration, original['permutations'])
    need(original['train_manifest_sha256'] == TRAIN_MANIFEST_SHA, 'Numeric TRAIN binding differs')
    tables = derive_tables(original['permutations'], v4['readout_permutations'])
    payload = dict(format=CANDIDATES_FORMAT, **base_binding(), train_token_hashes=v4['train_token_hashes'], tables=tables)
    args.out.mkdir(parents=True)
    path = args.out/'candidates.pt'
    write_payload(path, payload)
    receipt = dict(format=CANDIDATES_FORMAT, **base_binding(), train_token_hashes=v4['train_token_hashes'],
        complete=True, fresh_source_no_adapter=True, file=path.name, sha256=data.sha_file(path), bytes=path.stat().st_size,
        table_sha256={name:tensor_sha(table) for name, table in tables.items()}, runtime_table_bytes=57344,
        v4_calibration_receipt_sha256=data.sha_file(args.v4_calibration.with_suffix('.json')),
        collector_forward_exactcheck=v4_receipt['collection_forward_exactcheck'], code_sha256=code_hashes())
    write_new_json(path.with_suffix('.json'), receipt)
    load_candidates(path)
    need(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    print(json.dumps(dict(complete=True, candidates_sha256=receipt['sha256'], bytes=receipt['bytes'],
                         table_sha256=receipt['table_sha256'], cuda_initialized=False), indent=2))


if __name__ == '__main__':
    main()
