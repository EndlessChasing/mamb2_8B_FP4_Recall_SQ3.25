"""Direct identities for the frozen FP4 weight comparison; no state-search history."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = 'ecb0e811798c4da4c8e2a3d82fc3326d8abba3904bf920db13812384e3a5ec10'
WEIGHT_FORMATS = ('int4_control', 'fp4_g64_f16', 'fp4_g16_e4m3')
SOURCE_SHA = '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'
W4_MANIFEST_SHA = '3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
VALIDATION_TOKENS_SHA = '5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027'
TABLE_SHA = '214b47a4dfdc20fce4aa552f954e3f0af84b49edbfc14cbcc1569946a4777ef8'
SELECTION_SHA = '7ac1c824936adbcf372570f2f49a22b02a9d42f007e7ae506528e84392560fb2'
SELECTION_RECEIPT_SHA = '9f615b0c4265b6698d5880c7b651e37227b63cf38fbf38c73c3f31c694d408b3'
STATE_CHECKS_SHA = 'd9eb251a9c53f9713c362ca762fe6376a69f1b81470532eea6dfbb647b35023c'
STATE_AUDIT_SHA = 'cc031cd9b1bbe60969dd070730f1db6ba75c2ee139344323ca6b97eb1f8c72ca'
ARCHIVES = {
    's16': ('artifacts/pretrain_v1/pretrain_w4_s16.json',
            '09762a55056e5db0b587ea2c22fc9cd34a3edecc7aa3ccc5a6570522480bfc04', 8.014129751718814),
    'sq325': ('artifacts/repair_v5/full_v1/full_selected.json',
              'f57704010cc3cae530ba892e74f176563dd5f901615148217228f743aa284f63', 9.012334876908318),
}
LAYOUT = '32_32_64'
LAYOUTS = (LAYOUT,) * 56
CACHE_BYTES = 28499968
S16_CACHE_BYTES = 122028032
WEIGHT_BYTES = 16473999360
PARAMETER_COUNT = 8236999680
FP4_PARAMETER_COUNT = 8233418752
FP16_BYTES = 7161856
PAYLOAD_BYTES = {'fp4_g64_f16': 4381165568, 'fp4_g16_e4m3': 4638460360}
TARGET = 8.4
AUDIT_FORMAT = 'FP4_STATE_QUALITY_AUDIT_V1'
COMPARE_FORMAT = 'FP4_STATE_QUALITY_COMPARISON_V1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
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
    need(sha(ROOT / 'docs/FP4_WEIGHT_V1_PROTOCOL.md') == PROTOCOL_SHA, 'Frozen FP4 protocol changed')
    return PROTOCOL_SHA


def check_inventory(inventory):
    need(isinstance(inventory, dict) and inventory, 'Executed source inventory missing')
    for name, digest in inventory.items():
        path = (ROOT / name).resolve()
        need(not Path(name).is_absolute() and path.is_relative_to(ROOT.resolve())
             and path.is_file() and sha(path) == digest, 'Executed source changed: ' + name)


def code_hashes(extra=()):
    check_protocol()
    snapshots = read_json(ROOT / 'docs/UPSTREAM_SNAPSHOTS.json')
    paths = [name for name in snapshots if name.endswith('.py')]
    for name in paths:
        need(sha(ROOT / name) == snapshots[name]['sha256'], 'Inherited source changed: ' + name)
    paths += ['docs/UPSTREAM_SNAPSHOTS.json', 'docs/FP4_WEIGHT_V1_PROTOCOL.md',
              'docs/RESURFACE_MORE_BACKEND_REPLAY.md', 'mamba2_recall/fp4.py',
              'scripts/fp4_state_binding_v1.py', 'scripts/run_fp4_state_quality_v1.py',
              'scripts/run_w4_state_repair_v2.py', 'scripts/w4_state_binding_v1.py',
              'scripts/w4_state_repair_binding_v2.py', 'scripts/w4_state_repair_codec_v2.py', *extra]
    return {name: sha(ROOT / name) for name in sorted(set(paths))}


def file_receipt(path):
    path = Path(path)
    return dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size)


def require_audit(path, report_path, stage):
    need(path is not None, 'Independent CPU audit required: ' + stage)
    audit = read_json(path)
    need(audit.get('format') == AUDIT_FORMAT and audit.get('stage') == stage
         and audit.get('complete') is True and audit.get('passed') is True
         and audit.get('cuda_initialized') is False and 'error' not in audit
         and audit.get('protocol_sha256') == PROTOCOL_SHA
         and audit.get('input_report_sha256') == sha(report_path)
         and audit.get('source_sha256') == sha(ROOT / 'scripts/audit_fp4_state_quality_v1.py'),
         'Passing independently source-bound CPU audit required: ' + stage)
    return audit


def validate_codec_admission(checks_path, audit_path):
    checks = read_json(checks_path)
    need(checks.get('format') == 'FP4_WEIGHT_CODEC_CHECK_V1' and checks.get('complete') is True
         and checks.get('passed') is True and checks.get('cuda_initialized') is True
         and checks.get('protocol_sha256') == PROTOCOL_SHA and 'error' not in checks,
         'Passing bounded GPU FP4 codec fixtures required')
    check_inventory(checks['code_sha256'])
    need(checks['code_sha256'].get('mamba2_recall/fp4.py') == sha(ROOT / 'mamba2_recall/fp4.py')
         and checks['code_sha256'].get('scripts/check_fp4_weight_v1.py')
         == sha(ROOT / 'scripts/check_fp4_weight_v1.py'), 'FP4 fixtures do not bind current codec/checker')
    evidence = checks['evidence']
    need(Path(evidence['file']).name == evidence['file'], 'Unsafe codec evidence path')
    path = Path(checks_path).parent / evidence['file']
    need(sha(path) == evidence['sha256'] and path.stat().st_size == evidence['bytes'],
         'Actual codec evidence changed')
    require_audit(audit_path, checks_path, 'codec_checks')
    return dict(fp4_kernel_checks_sha256=sha(checks_path), fp4_kernel_audit_sha256=sha(audit_path))


def load_table(path):
    import torch
    from mamba2_recall.resurface_native import tensor_hash
    need(sha(path) == SELECTION_SHA and sha(Path(path).with_suffix('.json')) == SELECTION_RECEIPT_SHA,
         'Exact frozen top16 selection required')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    table = payload['permutations']
    need(payload.get('selected_id') == 'top16' and payload.get('selected_layout') == LAYOUT
         and payload.get('table_sha256') == TABLE_SHA and payload.get('adapter_used') is False
         and payload.get('heldout_used') is False and payload.get('mk_used') is False,
         'Frozen table metadata differs')
    need(table.dtype == torch.uint8 and list(table.shape) == [56, 8, 128]
         and table.is_contiguous() and tensor_hash(table) == TABLE_SHA
         and torch.equal(table.sort(-1).values, torch.arange(128, dtype=torch.uint8).expand_as(table)),
         'Actual frozen table bytes/permutations differ')
    return table


def load_archives():
    reports = {}
    for mode, (name, digest, ppl) in ARCHIVES.items():
        need(sha(ROOT / name) == digest, 'Archived INT4 control changed: ' + mode)
        report = read_json(ROOT / name)
        need(report.get('complete') is True and report.get('ppl_complete') is True
             and report['ppl']['ppl'] == ppl and len(report['ppl']['windows']) == 130
             and report['ppl']['target_tokens'] == 264764, 'Archived INT4 control incomplete')
        reports[mode] = report
    return reports


def validate_order(weight_format, control_report=None, control_audit=None, prior_report=None, prior_audit=None):
    need(weight_format in WEIGHT_FORMATS, 'Unknown weight format')
    if weight_format == 'int4_control':
        need(all(value is None for value in (control_report, control_audit, prior_report, prior_audit)),
             'INT4 control must start the fixed comparison order')
        return {}
    need(control_report is not None and control_audit is not None, 'Both FP4 formats require the audited INT4 control')
    require_audit(control_audit, control_report, 'full')
    control = read_json(control_report)
    need(control.get('format') == COMPARE_FORMAT and control.get('complete') is True
         and control.get('weight_format') == 'int4_control' and control.get('all_integrity_checks_passed') is True
         and control.get('protocol_sha256') == PROTOCOL_SHA and control.get('code_sha256') == code_hashes()
         and all(control.get('archived_replays', {}).get(mode, {}).get('complete') is True for mode in ARCHIVES)
         and all(control['metrics'][mode]['ppl'] == ARCHIVES[mode][2] for mode in ARCHIVES),
         'Exact completed INT4 control required before FP4 measurement')
    result = dict(int4_control_report_sha256=sha(control_report), int4_control_audit_sha256=sha(control_audit))
    if weight_format == 'fp4_g64_f16':
        need(prior_report is None and prior_audit is None, 'G64 follows the INT4 control directly')
    else:
        need(prior_report is not None and prior_audit is not None, 'G16 follows an independently audited G64 pair')
        require_audit(prior_audit, prior_report, 'full')
        prior = read_json(prior_report)
        need(prior.get('format') == COMPARE_FORMAT and prior.get('complete') is True
             and prior.get('weight_format') == 'fp4_g64_f16' and prior.get('all_integrity_checks_passed') is True
             and prior.get('protocol_sha256') == PROTOCOL_SHA and prior.get('code_sha256') == code_hashes()
             and all(prior['input_binding'].get(key) == value for key, value in result.items()),
             'G64 predecessor integrity or shared control differs')
        result.update(prior_report_sha256=sha(prior_report), prior_audit_sha256=sha(prior_audit))
    return result
