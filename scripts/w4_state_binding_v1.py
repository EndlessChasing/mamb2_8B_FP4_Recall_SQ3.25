"""Explicit W4/state experiment identities; no rebinding of old experiments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall import runtime, w4, resurface_native as native

PROTOCOL_SHA = 'b75b607ff9a8a1af1a3a5c478a9b98d95565efc23dc28a453ccd2a41a5b698ab'
W4_MANIFEST_SHA = '3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05'
TRAIN_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
NUMERIC_PROTOCOL_SHA = '24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb'
TRAIN_MANIFEST_SHA = '451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0'
CONFIRM_CASE_SHA = '445f1ebae5cfeeafccc0b6c02f1a514d8d7858360637af9ec37ab5e48212a091'
VALIDATION_TOKENS_SHA = '5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027'
LAYOUT = '32_32_64'
CACHE_BYTES = 28499968
S16_CACHE_BYTES = 56 * 128 * 64 * 128 * 2 + 4587520
ADAPTER_BYTES = 2308208
CALIBRATION_FORMAT = 'W4_STATE_CALIBRATION_V1'
CANDIDATE_ORDER = ('transferred_fp16_v10', 'w4_magnitude', 'w4_readout', 'w4_preserve32', 'w4_preserve64')
CALIBRATION_ROWS = tuple(range(216,224))
SCREEN_ROWS = tuple(range(224,256))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def check_protocol():
    need(sha(ROOT/'docs/W4_STATE_RESURFACE_V1_PROTOCOL.md') == PROTOCOL_SHA,
         'New W4/state prospective protocol changed')
    need(sha(ROOT/'docs/QUANT_FIRST_PROTOCOL.md') == NUMERIC_PROTOCOL_SHA,
         'Inherited numeric-data protocol changed')
    return PROTOCOL_SHA


def code_hashes(extra=()):
    origins = read_json(ROOT/'docs/UPSTREAM_SNAPSHOTS.json')
    paths = [name for name in origins if name.endswith('.py')]
    for name in paths:
        need(sha(ROOT/name) == origins[name]['sha256'], 'Inherited numerical/helper source changed: '+name)
    paths += ['scripts/w4_state_binding_v1.py', 'docs/W4_STATE_RESURFACE_V1_PROTOCOL.md',
              'docs/UPSTREAM_SNAPSHOTS.json'] + list(extra)
    return {name: sha(ROOT/name) for name in sorted(set(paths))}


def base_manifest(w4_dir):
    path = Path(w4_dir)/'manifest.json'
    need(sha(path) == W4_MANIFEST_SHA, 'Exact independently published W4 manifest required')
    manifest = read_json(path)
    need(manifest['complete'] is True and len(manifest['tensors']) == 507,
         'Incomplete W4 package')
    need(manifest['source_checkpoint_sha256'] == runtime.SOURCE_CHECKPOINT_SHA256,
         'W4 original checkpoint provenance differs')
    return manifest


def expected_base_hashes(w4_dir):
    return {name: entry['decoded_sha256'] for name,entry in base_manifest(w4_dir)['tensors'].items()}


def load_w4(w4_dir):
    check_protocol()
    base_manifest(w4_dir)
    model = w4.load_w4_model(w4_dir)
    need(model._package_receipt['manifest_sha256'] == W4_MANIFEST_SHA
         and model._package_receipt['file_hashes_verified'] is True
         and model._package_receipt['decoded_hashes_verified'] is True,
         'W4 loader did not verify actual files and decoded values')
    need(not any(mx._forward_pre_hooks or mx._forward_hooks or mx.norm._forward_pre_hooks
                 for mx in (layer.mixer for layer in model.backbone.layers)),
         'The new W4 base must have no adapter hooks')
    return model


def assert_loaded_hashes(model, expected):
    values = model.state_dict()
    need(set(values) == set(expected), 'Actual loaded W4 tensor inventory differs')
    actual = {}
    for name,value in values.items():
        need(value.dtype == torch.float16 and not value.is_meta, 'Unexpected loaded W4 dtype')
        actual[name] = native.tensor_hash(value)
        need(actual[name] == expected[name], 'Actual loaded W4 bytes changed: '+name)
    return dict(complete=True, passed=True, actual_content_checked=True,
                tensors=len(actual), decoded_tensor_sha256=actual,
                weight_payload_bytes=sum(t.numel()*t.element_size() for t in values.values()))


def check_table(table):
    need(isinstance(table,torch.Tensor) and table.device.type == 'cpu'
         and table.dtype == torch.uint8 and tuple(table.shape) == (56,8,128),
         'Expected uint8 CPU coordinate table [56,8,128]')
    need(torch.equal(table.sort(-1).values, torch.arange(128,dtype=torch.uint8).expand_as(table)),
         'State table rows must be permutations of 0..127')
    return native.tensor_hash(table)


def load_selection(calibration):
    check_protocol()
    calibration = Path(calibration)
    payload = torch.load(calibration, map_location='cpu', weights_only=True)
    receipt = read_json(calibration.with_suffix('.json'))
    expected = dict(format=CALIBRATION_FORMAT, protocol_sha256=PROTOCOL_SHA,
                    w4_manifest_sha256=W4_MANIFEST_SHA, selected_layout=LAYOUT,
                    scale_mode='stored_scale', int4_clip=1., adapter_used=False,
                    heldout_used=False, mk_used=False, train_file_sha256=TRAIN_SHA)
    need(all(payload.get(k) == v and receipt.get(k) == v for k,v in expected.items()),
         'Selected W4/state artifact provenance differs')
    need(receipt.get('complete') is True and receipt['sha256'] == sha(calibration)
         and receipt['bytes'] == calibration.stat().st_size, 'Selected state payload differs')
    table_sha = check_table(payload['permutations'])
    need(table_sha == payload['table_sha256'] == receipt['table_sha256'], 'Selected table hash differs')
    need(payload['selected_id'] == receipt['selected_id'] and payload['selected_id'] in CANDIDATE_ORDER,
         'Selected state candidate is outside the declared grid')
    screen_path = calibration.parent/'screen_comparison.json'
    need(sha(screen_path) == payload['selection_report_sha256'] == receipt['selection_report_sha256'],
         'Selected TRAIN comparison changed')
    comparison = read_json(screen_path)
    need(comparison.get('complete') is True, 'Incomplete TRAIN comparison')
    need(comparison.get('format') == 'W4_STATE_SCREEN_V1'
         and comparison.get('protocol_sha256') == PROTOCOL_SHA
         and comparison.get('w4_manifest_sha256') == W4_MANIFEST_SHA
         and tuple(comparison.get('candidate_order', [])) == CANDIDATE_ORDER
         and comparison.get('selection', {}).get('selected_id') == payload['selected_id']
         and comparison.get('selection', {}).get('selected_layout') == LAYOUT
         and comparison.get('table_sha256', {}).get(payload['selected_id']) == table_sha,
         'Selected payload and frozen TRAIN comparison disagree')
    binding = dict(protocol_sha256=PROTOCOL_SHA, w4_manifest_sha256=W4_MANIFEST_SHA,
                   source_checkpoint_sha256=runtime.SOURCE_CHECKPOINT_SHA256,
                   tokenizer_sha256=runtime.TOKENIZER_SHA256,
                   calibration_sha256=sha(calibration),
                   calibration_receipt_sha256=sha(calibration.with_suffix('.json')),
                   calibration_format=CALIBRATION_FORMAT, layout=LAYOUT,
                   table_sha256=table_sha, selected_id=payload['selected_id'],
                   selection_report_sha256=payload['selection_report_sha256'])
    return payload,receipt,binding


def validate_upstream_kernels():
    directory = ROOT/'reference/sq325'
    pins = {
        'state_resurface_v11_training_checks_attempt1.json': '31fa24e451e946bb8d7f1da53bf888591e98ec3e0c19063013180a6514f5b594',
        'state_resurface_v11_training_checks_attempt1_evidence.pt': 'fdaf2f7e8ac105ba2877b48e15382f3abe59ee5596edb6dc6e37ae36bd0c4480',
        'state_resurface_v11_checks_audit.json': '439c6fd8f5aed2a0983277a4cc5064ea6f7f894b502cad40175321dfc16b506b',
        'state_repair_collector_checks.json': '6c6134a8d08ade38f1aa8ca331c677bc6d156e9bd56cb68930e98f80f77f5b5b'}
    for name,digest in pins.items():
        need(sha(directory/name) == digest, 'Upstream fixture/evidence changed: '+name)
    checks = read_json(directory/'state_resurface_v11_training_checks_attempt1.json')
    audit = read_json(directory/'state_resurface_v11_checks_audit.json')
    collector = read_json(directory/'state_repair_collector_checks.json')
    for report in (checks,audit,collector):
        need(report.get('complete') is True and report.get('passed') is True,
             'Passing upstream numerical checks required')
    need(audit['checks_sha256'] == pins['state_resurface_v11_training_checks_attempt1.json']
         and audit['kernel_checks']['evidence_sha256'] == pins['state_resurface_v11_training_checks_attempt1_evidence.pt']
         and audit['kernel_checks']['checks'] == 189 and audit['cuda_initialized'] is False,
         'Upstream independent fixture audit binding differs')
    for report in (checks,collector):
        for name,digest in report['code_sha256'].items():
            need(sha(ROOT/name) == digest, 'Reused numerical source changed: '+name)
    return dict(complete=True, passed=True, reused_prior_checks=True,
                new_gpu_fixture_run=False, numerical_sources_unchanged=True,
                checks=189, raw_cases=9, fixture_and_audit_sha256=pins,
                scope='Exact inherited kernel/STE/collector fixtures; W4 model integration is checked separately.')
