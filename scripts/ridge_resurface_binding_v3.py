"""Frozen ridge-state parent and provenance for Resurface V3 experiments."""
from pathlib import Path
import torch
from prepare_state_first_v5 import tensor_sha
import fp4_state_binding_v1 as fp4

ROOT = Path(__file__).resolve().parents[1]
need, sha, read, put = fp4.need, fp4.sha, fp4.read_json, fp4.write_json
PARENT_PPL = 8.408282583578627
CACHE_BYTES = 28499968
ADAPTER_BYTES = 2308208
PROTOCOL = ROOT / 'docs/FP4_G16_RIDGE_RESURFACE_V3_PROTOCOL.md'
ARCHIVE = ROOT / 'artifacts/fp4_weight_v1/fp4_g16_v1'
FILES = ('scripts/ridge_resurface_binding_v3.py',
         'scripts/run_fp4_ridge_resurface_v3.py',
         'scripts/audit_fp4_ridge_resurface_v3.py',
         'scripts/fp4_zero_predictor_codec_v1.py',
         'scripts/run_fp4_latent_state_quality_v1.py',
         'scripts/run_w4_state_resurface_v1.py',
         'mamba2_recall/resurface_native.py', 'mamba2_recall/fp4.py',
         'docs/FP4_G16_RIDGE_RESURFACE_V3_PROTOCOL.md')


def hashes(extra=()):
    return {name: sha(ROOT / name) for name in (*FILES, *extra)}


def add_parent_arguments(parser):
    defaults = dict(static_dir='artifacts/fp4_zero_predictor_v1/group_ridge_static_v1',
                    ridge_full_dir='artifacts/fp4_zero_predictor_v1/group_ridge_full_v1',
                    top4_rank_dir='artifacts/fp4_state_layer_v1/rank_v1',
                    top4_full_dir='artifacts/fp4_state_layer_v1/full_v1')
    for key, value in defaults.items():
        parser.add_argument('--' + key.replace('_','-'), type=Path, default=ROOT / value)


def load_parent(args):
    static = read(args.static_dir / 'report.json')
    sa = read(args.static_dir / 'audit.json')
    comp = read(args.ridge_full_dir / 'comparison.json')
    audit = read(args.ridge_full_dir / 'audit.json')
    saved = read(args.ridge_full_dir / 'replacement.json')
    need(static['complete'] is True and sa['passed'] is True and
         sa['cuda_initialized'] is False and
         sa['input_report_sha256'] == sha(args.static_dir / 'report.json') and
         static['static_sha256'] == sha(args.static_dir / 'static.pt') and
         comp['complete'] is True and comp['stage'] == 'full' and
         comp['selection']['selected_id'] == 'replacement' and
         comp['ppl']['replacement'] == PARENT_PPL and audit['passed'] is True and
         audit['cuda_initialized'] is False and
         audit['input_report_sha256'] == sha(args.ridge_full_dir / 'comparison.json') and
         comp['report_sha256']['replacement'] == sha(args.ridge_full_dir / 'replacement.json') and
         saved['ppl']['ppl'] == PARENT_PPL, 'Audited ridge parent differs')
    rank = read(args.top4_rank_dir / 'comparison.json')
    top4 = read(args.top4_full_dir / 'comparison.json')
    ta = read(args.top4_full_dir / 'audit.json')
    need(rank['combos_sha256'] == sha(args.top4_rank_dir / 'combos.pt') and
         ta['passed'] is True and ta['cuda_initialized'] is False and
         ta['input_report_sha256'] == sha(args.top4_full_dir / 'comparison.json'),
         'Audited frozen top4 table differs')
    table = torch.load(args.top4_rank_dir / 'combos.pt', map_location='cpu',
                       weights_only=True)['tables']['top4'].contiguous()
    table_sha = tensor_sha(table)
    need(table.dtype == torch.uint8 and tuple(table.shape) == (56,8,128) and
         torch.equal(table.sort(-1).values, torch.arange(128,dtype=torch.uint8).expand_as(table)) and
         table_sha == top4['table_sha256']['top4'] == saved['table_sha256'], 'Table differs')
    payload = torch.load(args.static_dir / 'static.pt', map_location='cpu', weights_only=True)
    need(payload['format'] == 'FP4_GROUP_RIDGE_STATIC_V1' and
         tuple(payload['layouts']) == tuple(saved['layer_layouts']), 'Static format/layout differs')
    bases, scales, predictors = (payload[k] for k in ('bases','scales','replacement_predictor'))
    digest = dict(bases=[tensor_sha(x) for x in bases], scales=[tensor_sha(x) for x in scales],
                  predictors=[[tensor_sha(x) for x in row] for row in predictors])
    need(digest == static['replacement_static_sha256'] and
         digest['bases'] == saved['static_basis_sha256'] and
         digest['scales'] == saved['static_scale_sha256'] and
         digest['predictors'] == saved['static_predictor_sha256'], 'Static values differ')
    stacked = [torch.stack(row).contiguous() for row in predictors]
    need(sum(x.numel()*2 for x in bases) == 109952 and
         sum(x.numel()*2 for x in scales) == 1792 and
         sum(x.numel()*2 for x in stacked) == 3673216, 'Static physical bytes differ')
    expected = {k:v['decoded_sha256'] for k,v in read(ARCHIVE / 'conversion_receipt.json')['tensors'].items()}
    provenance = dict(parent_comparison_sha256=sha(args.ridge_full_dir / 'comparison.json'),
        parent_audit_sha256=sha(args.ridge_full_dir / 'audit.json'),
        parent_arm_sha256=sha(args.ridge_full_dir / 'replacement.json'),
        parent_ppl=PARENT_PPL, static_payload_sha256=sha(args.static_dir / 'static.pt'),
        static_report_sha256=sha(args.static_dir / 'report.json'),
        table_payload_sha256=sha(args.top4_rank_dir / 'combos.pt'),
        table_sha256=table_sha, static_tensor_sha256=digest,
        layouts=list(payload['layouts']), source_checkpoint_sha256=fp4.SOURCE_SHA,
        conversion_receipt_sha256=sha(ARCHIVE / 'conversion_receipt.json'))
    return table, tuple(payload['layouts']), bases, scales, stacked, saved, expected, provenance
