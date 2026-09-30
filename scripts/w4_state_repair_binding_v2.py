"""Separate identities for the unadapted W4/Q3.25 PPL <8.4 experiment."""
from pathlib import Path
import w4_state_binding_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '1ef9f9734e1321c6331aac6fb61d49c7a65859bf6225ae3062839c03c95eecc3'
RAW_STATS_SHA = '742cf9c2e8b7047b73338a0902287123872a026028fde8fcdcf3d9a68e2d6a82'
BASELINE_REPORT_SHA = 'e4bec0537c27cbf61cabee6bbcf1508882cbac487060b088f7638c6cffc5ceed'
BASELINE_PPL = 9.103235516433815
S16_PPL = 8.014129751718814
TARGET = 8.4
LAYOUT_ORDER = ('32_32_64','16_64_48','8_80_40','24_48_56',
                '36_24_68','40_16_72','44_8_76')
TABLE_ORDER = parent.CANDIDATE_ORDER
BASELINE_ID = '32_32_64__transferred_fp16_v10'
SCREEN_ROWS = tuple(range(256,288))
W4_MANIFEST_SHA = parent.W4_MANIFEST_SHA
TRAIN_SHA = parent.TRAIN_SHA
VALIDATION_TOKENS_SHA = parent.VALIDATION_TOKENS_SHA
CACHE_BYTES = parent.CACHE_BYTES
need, sha, read_json, write_json = parent.need, parent.sha, parent.read_json, parent.write_json


def check_protocol():
    parent.check_protocol()
    need(sha(ROOT/'docs/W4_STATE_REPAIR_V2_PROTOCOL.md') == PROTOCOL_SHA,
         'The prospective unadapted W4 PPL-repair protocol changed')
    return PROTOCOL_SHA


def code_hashes(extra=()):
    check_protocol()
    return parent.code_hashes(extra=(
        'scripts/w4_state_repair_binding_v2.py',
        'scripts/w4_state_repair_codec_v2.py',
        'docs/W4_STATE_REPAIR_V2_PROTOCOL.md', *extra))
