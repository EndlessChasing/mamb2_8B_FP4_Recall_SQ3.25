#!/usr/bin/env python3
"""TRAIN-only group interventions around the frozen v8 TRAIN winner."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mamba2_recall import runtime, resurface_data as data
from prepare_quant_first import load_train_tokens, write_new_json
from prepare_state_first_v5 import KINDS, need, read_json, tensor_sha, check_table, write_payload
import prepare_state_ppl_v8 as p8
import run_state_ppl_v8 as r8
import run_state_ppl_v6 as v6

PROTOCOL_SHA = '9e01c03ee6870a8ecbcd9a0ba9157b1651d2830ea65d2951664ebcb81b4a09b3'
V8_CALIBRATION_SHA = 'e7fa9ce0f245d3843f447be604bc195186645cb956f9d229cf34b1d05a9bdc49'
V8_SELECTED_SHA = '9b7c04814085abbb67d7a10e0b70d1a6e9b34e97299fc0dc26e4373ea8f8ea39'
V8_SCREEN_SHA = 'ac3604395393d0dccda8e299021ce686653a02babbeaba33bfe58d6d4ba7b16c'
V8_SCREEN_AUDIT_SHA = '90fbed838e0c97f5543b6687b20d6a3740fc4fa0ebade2920d73d08df40526e2'
LAYERS = (0,3,6,1,4,2,7,10)
CALIBRATION_ROWS = tuple(range(144,152))
SCREEN_ROWS = tuple(range(152,184))
TOP_K = (1,2,4,8,16,32)
FORMAT = 'MAMBA2_STATE_PPL_GROUP_EVAL_V1'
COMPARE = 'MAMBA2_STATE_PPL_GROUP_CALIBRATION_V1'
CANDIDATES = 'MAMBA2_STATE_PPL_GROUP_CANDIDATES_V1'
INPUT_NAMES = ('source-dir','candidates','v5-selected-calibration','v6-selected-calibration',
    'prose-tokens','codec-checks','layer-candidates','v8-selected-calibration','v8-screen-audit','v8-full-dir','v8-full-audit')


def code_hashes():
    result = r8.code_hashes()
    for name in ('scripts/prepare_state_ppl_v9.py','scripts/audit_state_ppl_v9.py','docs/STATE_PPL_V9_PROTOCOL.md'):
        result[name] = data.sha_file(ROOT/name)
    return dict(sorted(result.items()))


def check_protocol():
    need(data.sha_file(ROOT/'docs/STATE_PPL_V9_PROTOCOL.md') == PROTOCOL_SHA, 'Frozen v9 protocol differs')


def validate_parent_miss(args, parent):
    """Require the independently audited finite v8 miss, not a failed process."""
    directory = args.v8_full_dir
    comp_path = directory/'full_comparison.json'
    comp = read_json(comp_path)
    audit = read_json(args.v8_full_audit)
    need(audit.get('format') == 'MAMBA2_STATE_PPL_V8_INDEPENDENT_AUDIT_V1'
         and audit.get('complete') is True and audit.get('passed') is True
         and audit.get('cuda_initialized') is False and audit.get('stage') == 'full'
         and audit.get('protocol_sha256') == p8.PROTOCOL_SHA
         and audit.get('source_sha256') == data.sha_file(ROOT/'scripts/audit_state_ppl_v8.py'),
         'A passing independent v8 full audit is required')
    dependencies = {'audit_state_ppl_v6.py','audit_state_ppl_v7.py','audit_quant_first.py',
        'audit_state_repair.py','audit_state_first_v5.py','audit_resurface_more.py'}
    need(set(audit['auditor_dependency_sha256']) == dependencies and all(
        data.sha_file(ROOT/'scripts'/name) == digest for name,digest in audit['auditor_dependency_sha256'].items()),
        'v8 auditor source dependency differs')
    proof = audit['full']
    selected_sha = data.sha_file(args.v8_selected_calibration)
    need(comp.get('format') == r8.COMPARE and comp.get('complete') is True and comp.get('stage') == 'full'
         and comp.get('protocol_sha256') == p8.PROTOCOL_SHA and comp.get('code_hashes') == r8.code_hashes()
         and comp.get('adapter_loaded') is False and comp.get('adapter_sha256') is None
         and comp.get('mk_used') is False and comp.get('resurface_trained') is False
         and comp.get('selected_id') == parent['selected_id'] == 'top8'
         and comp.get('selected_calibration_sha256') == selected_sha
         and comp.get('selection_report_sha256') == parent['selection_report_sha256']
         and audit['selected_calibration']['sha256'] == selected_sha
         and proof.get('complete') is True and proof['comparison_sha256'] == data.sha_file(comp_path)
         and proof['report_sha256'] == comp['report_sha256'], 'v8 outcome does not bind the frozen TRAIN parent')
    need(tuple(comp['report_sha256']) == r8.FULL_ARMS, 'v8 full arm inventory differs')
    for arm,digest in comp['report_sha256'].items():
        need(data.sha_file(directory/('full_'+arm+'.json')) == digest, 'Audited v8 raw arm changed')
    need(comp.get('target_pass') is False and proof.get('target_pass') is False
         and comp['target_checks'] == proof['target_checks'] == dict(ppl_strictly_below_8p25=False,
             cache_same_budget=True,all_integrity_checks_passed=True),
         'v8 target passed or integrity failed; do not execute v9')
    selected = read_json(directory/'full_selected.json')
    need(v6.valid_candidate(selected) and selected['candidate_table_sha256'] == parent['table_sha256']
         and selected['selected_calibration_sha256'] == selected_sha
         and math.isfinite(selected['ppl']['ppl']) and selected['ppl']['ppl'] >= 8.25,
         'v8 parent must be a finite same-budget target miss')
    return selected, dict(complete=True,comparison_sha256=data.sha_file(comp_path),
        audit_sha256=data.sha_file(args.v8_full_audit),parent_report_sha256=data.sha_file(directory/'full_selected.json'),
        selected_calibration_sha256=selected_sha,parent_ppl=selected['ppl']['ppl'],
        parent_selection='Frozen v8 TRAIN top8, irrespective of validation comparison',
        criterion='Independent full audit passed; finite v8 PPL >=8.25; all integrity guards passed')


def group_inventory(tables, parent):
    """Skip parent-equal bytes and deduplicate alternatives in frozen KINDS order."""
    check_table(parent)
    inventory = []
    for layer in LAYERS:
        for group in range(8):
            parent_group = parent[layer,group]
            seen = {}
            alternatives = []
            for kind in KINDS:
                actual = tables[kind][layer,group]
                raw = actual.numpy().tobytes()
                arm = f'layer{layer:02d}_group{group}_{kind}'
                if torch.equal(actual,parent_group):
                    disposition, duplicate, kept_arm = 'parent_equal', None, None
                elif raw in seen:
                    disposition, duplicate, kept_arm = 'duplicate_of', seen[raw], None
                else:
                    disposition, duplicate, kept_arm = 'retained', None, arm
                    seen[raw] = kind
                alternatives.append(dict(kind=kind,group_sha256=tensor_sha(actual),
                    disposition=disposition,duplicate_of=duplicate,arm=kept_arm))
            need(any(row['disposition'] == 'parent_equal' for row in alternatives),
                 'Parent group must come from an existing v5 table')
            inventory.append(dict(layer=layer,group=group,parent_group_sha256=tensor_sha(parent_group),alternatives=alternatives))
    return inventory


def arm_specs(inventory):
    return [('baseline',None,None,None)] + [(r['arm'],row['layer'],row['group'],r['kind'])
        for row in inventory for r in row['alternatives'] if r['disposition'] == 'retained'] + [('restored_baseline',None,None,None)]


def load_inputs(args):
    check_protocol()
    v8_candidates, v8_binding, v8_windows = r8.load_inputs(args)
    need(v8_candidates['calibration_report_sha256'] == V8_CALIBRATION_SHA, 'Fixed eligible-layer evidence differs')
    v8_cal = read_json(args.layer_candidates.parent/'calibration_comparison.json')
    need(tuple(row['layer'] for row in v8_cal['ranking']['ranked_improving_swaps'][:8]) == LAYERS,
         'Fixed eligible-layer order differs')
    parent, _, _ = r8.load_selection(args.v8_selected_calibration,v8_candidates,v8_binding,v8_windows)
    need(parent['selected_id'] == 'top8' and not parent['baseline_wins'], 'Parent must remain v8 TRAIN top8')
    need(data.sha_file(args.v8_selected_calibration) == V8_SELECTED_SHA
         and parent['selection_report_sha256'] == V8_SCREEN_SHA
         and data.sha_file(args.v8_screen_audit) == V8_SCREEN_AUDIT_SHA,'Pinned v8 TRAIN selection/audit differs')
    screen_audit = read_json(args.v8_screen_audit)
    need(screen_audit.get('complete') is True and screen_audit.get('passed') is True
         and screen_audit.get('stage') == 'screen' and screen_audit.get('cuda_initialized') is False
         and screen_audit['screen']['comparison_sha256'] == V8_SCREEN_SHA
         and screen_audit['selected_calibration']['sha256'] == V8_SELECTED_SHA,
         'Passing independent v8 screen audit required')
    archived, start = validate_parent_miss(args,parent)
    original, _ = p8.v6.load_candidates(args.candidates)
    inventory = group_inventory(original['tables'],parent['permutations'])
    need(len(arm_specs(inventory)) <= 194, 'Intervention inventory exceeds frozen limit')
    binding = dict(protocol_sha256=PROTOCOL_SHA,v8_input_binding=v8_binding,
        v8_selected_calibration_sha256=data.sha_file(args.v8_selected_calibration),
        v8_selected_receipt_sha256=data.sha_file(args.v8_selected_calibration.with_suffix('.json')),
        v8_screen_audit_sha256=V8_SCREEN_AUDIT_SHA,
        v8_selection_report_sha256=parent['selection_report_sha256'],
        v8_calibration_report_sha256=V8_CALIBRATION_SHA,parent_table_sha256=parent['table_sha256'],
        parent_selected_id='top8',start_condition=start,source_sha256=runtime.SOURCE_CHECKPOINT_SHA256,
        tokenizer_sha256=runtime.TOKENIZER_SHA256,train_file_sha256=v6.TRAIN_SHA)
    train = load_train_tokens(args.prose_tokens)
    windows = [(row*2048,train[row].clone()) for row in CALIBRATION_ROWS]
    return original,parent,binding,windows,inventory,archived


def table_for_arm(tables,parent,layer,group,kind):
    table = parent.clone().contiguous()
    if layer is None:
        need(group is None and kind is None, 'Invalid baseline replacement')
    else:
        need(layer in LAYERS and type(group) is int and 0 <= group < 8 and kind in KINDS,'Invalid group intervention')
        table[layer,group].copy_(tables[kind][layer,group])
    check_table(table)
    return table


def execution_spec(name):
    spec = v6.candidate_spec(p8.BASELINE_POLICY)
    spec.update(candidate_id=name,candidate_name='static_group_mix')
    return spec


def rank_groups(rows,inventory):
    need(v6.valid_candidate(rows['baseline']), 'Valid group-calibration parent required')
    nll0 = rows['baseline']['ppl']['nll']
    groups = []
    for entry in inventory:
        options = []
        for alt in entry['alternatives']:
            if alt['disposition'] != 'retained':continue
            row = rows[alt['arm']]; valid = v6.valid_candidate(row)
            options.append(dict(arm=alt['arm'],kind=alt['kind'],valid=valid,
                nll=row['ppl']['nll'] if valid else None,delta_nll=row['ppl']['nll']-nll0 if valid else None,
                ppl=row['ppl']['ppl'] if valid else None))
        valid = [row for row in options if row['valid']]
        best = min(valid,key=lambda row:(row['nll'],KINDS.index(row['kind']))) if valid else None
        groups.append(dict(layer=entry['layer'],group=entry['group'],alternatives=options,best_alternative=best,
            improving=best is not None and best['nll'] < nll0))
    ranked = [dict(layer=row['layer'],group=row['group'],**row['best_alternative']) for row in groups if row['improving']]
    ranked.sort(key=lambda row:(row['delta_nll'],row['layer'],row['group'],KINDS.index(row['kind'])))
    return dict(baseline_nll=nll0,baseline_ppl=rows['baseline']['ppl']['ppl'],groups=groups,
        ranked_improving_swaps=ranked,improving_groups=len(ranked),stopped=not ranked,
        calibration_rows=list(CALIBRATION_ROWS),future_screen_rows=list(SCREEN_ROWS),
        rule='Per-group minimum raw NLL; fixed v5 table order ties; strict improvement; global(deltaNLL,layer,group,table order)',
        interpretation='Individual group deltas propose combinations; effects are not assumed additive',
        adapter_used=False,heldout_used=False,mk_used=False)


def derive_candidates(tables,parent,ranking):
    ranked = ranking['ranked_improving_swaps']
    requests = [('baseline',0)]+[(f'top{k}',min(k,len(ranked))) for k in TOP_K]+[('allnegative',len(ranked))]
    unique,specs,seen = {},{},{}
    proposal_rows = []
    for name,count in requests:
        table = parent.clone().contiguous()
        swaps = [dict(layer=r['layer'],group=r['group'],kind=r['kind'],calibration_delta_nll=r['delta_nll'],
            calibration_arm=r['arm']) for r in ranked[:count]]
        for swap in swaps:table[swap['layer'],swap['group']].copy_(tables[swap['kind']][swap['layer'],swap['group']])
        check_table(table); digest = tensor_sha(table); kept = seen.get(digest)
        if kept is None:
            kept = name; seen[digest] = name; unique[name] = table
            specs[name] = dict(table_sha256=digest,swap_count=count,swaps=swaps,scale_mode='stored_scale',int4_clip=1.,cache_bytes=v6.CACHE_BYTES)
        else:need(torch.equal(unique[kept],table),'Whole-table hash collision')
        proposal_rows.append(dict(requested_name=name,requested_top_k='all' if name == 'allnegative' else 0 if name == 'baseline' else int(name[3:]),
            effective_swap_count=count,retained_name=kept,duplicate_of=kept if kept != name else None,table_sha256=digest))
    return unique,specs,proposal_rows


def dataset():
    return dict(split='train',file_sha256=v6.TRAIN_SHA,rows=list(CALIBRATION_ROWS),tokens_per_row=2048,windows=8,target_tokens=16376)


def candidate_metadata(binding,inventory,ranking,comparison_path,tables,specs,proposals):
    return dict(protocol_sha256=PROTOCOL_SHA,input_binding=binding,intervention_inventory=inventory,
        calibration_report_sha256=data.sha_file(comparison_path),candidate_order=list(tables),candidate_specs=specs,
        proposals=proposals,calibration_stopped=ranking['stopped'],baseline_name='baseline',scale_mode='stored_scale',int4_clip=1.,
        runtime_table_bytes=57344,cache_bytes=v6.CACHE_BYTES,adapter_loaded=False,adapter_sha256=None,heldout_used=False,mk_used=False)


def validate_calibration(directory,original,parent,binding,windows,inventory):
    path = directory/'calibration_comparison.json'; comp = read_json(path); specs = arm_specs(inventory)
    need(comp.get('format') == COMPARE and comp.get('complete') is True and comp.get('stage') == 'calibration'
         and comp.get('protocol_sha256') == PROTOCOL_SHA and comp.get('input_binding') == binding
         and comp.get('code_hashes') == code_hashes() and comp.get('intervention_inventory') == inventory
         and comp.get('adapter_loaded') is False and comp.get('adapter_sha256') is None
         and comp.get('heldout_used') is False and comp.get('mk_used') is False
         and tuple(comp['report_sha256']) == tuple(row[0] for row in specs), 'Group calibration comparison binding differs')
    inventory_path = directory/'intervention_inventory.json'
    need(data.sha_file(inventory_path) == comp['intervention_inventory_sha256']
         and read_json(inventory_path) == dict(complete=True,input_binding=binding,
             protocol_sha256=PROTOCOL_SHA,intervention_inventory=inventory,arms=[row[0] for row in specs],code_hashes=code_hashes()),
         'Premodel intervention inventory differs')
    rows = {}; identities = v6.window_identity(windows)
    for arm,layer,group,kind in specs:
        raw_path = directory/('calibration_'+arm+'.json')
        need(data.sha_file(raw_path) == comp['report_sha256'][arm], 'Raw group intervention changed')
        row = read_json(raw_path); table = table_for_arm(original['tables'],parent['permutations'],layer,group,kind)
        expected = dict(format=FORMAT,stage='calibration',arm=arm,protocol_sha256=PROTOCOL_SHA,input_binding=binding,
            dataset=dataset(),code_hashes=code_hashes(),candidate_table_sha256=tensor_sha(table),
            group_replacement=dict(layer=layer,group=group,kind=kind),adapter_loaded=False,adapter_sha256=None,
            heldout_used=False,heldout_used_for_selection=False,mk_used=False,**execution_spec(arm))
        need(all(row.get(key) == value for key,value in expected.items()), 'Group intervention identity differs')
        if row.get('complete'):
            v6.check_ppl(row,identities);need(v6.valid_candidate(row),'Complete group intervention failed integrity')
        else:need(layer is not None and row.get('excluded_from_selection') is True
            and row.get('error_type') == 'CandidateInvalid' and not row.get('fatal_failure',False),'Unrecognized group failure')
        rows[arm] = row
    ranking = rank_groups(rows,inventory); restoration = v6.check_restoration(rows['baseline'],rows['restored_baseline'])
    need(comp['ranking'] == ranking and comp['restoration'] == restoration
         and read_json(directory/'calibration_restoration.json') == restoration,'Group ranking/restoration differs')
    return comp,ranking


def load_group_candidates(path,args):
    original,parent,binding,windows,inventory,archive = load_inputs(args)
    comp,ranking = validate_calibration(path.parent,original,parent,binding,windows,inventory)
    tables,specs,proposals = derive_candidates(original['tables'],parent['permutations'],ranking)
    payload = torch.load(path,map_location='cpu',weights_only=True); receipt = read_json(path.with_suffix('.json'))
    metadata = candidate_metadata(binding,inventory,ranking,path.parent/'calibration_comparison.json',tables,specs,proposals)
    need(payload.get('format') == receipt.get('format') == CANDIDATES and receipt.get('complete') is True
         and receipt.get('file') == path.name and receipt.get('sha256') == data.sha_file(path)
         and receipt.get('bytes') == path.stat().st_size and receipt.get('code_hashes') == code_hashes()
         and all(payload.get(k) == receipt.get(k) == val for k,val in metadata.items())
         and tuple(payload['tables']) == tuple(tables),'Group candidates export/provenance differs')
    for name,table in tables.items():
        check_table(payload['tables'][name]);need(torch.equal(payload['tables'][name],table),'Group candidate table differs')
    return payload,receipt,comp,archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUT_NAMES,'out'):parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args();need(not args.out.exists(),'Fresh output directory required')
    original,parent,binding,windows,inventory,_ = load_inputs(args)
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256 == runtime.TOKENIZER_SHA256,'Tokenizer differs')
    torch.set_num_threads(8);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    policy = v6.pin_replay_backend();args.out.mkdir(parents=True)
    model = runtime.load_source_model(args.source_dir);frozen = v6.FrozenBase(model)
    common = dict(format=FORMAT,stage='calibration',protocol_sha256=PROTOCOL_SHA,input_binding=binding,dataset=dataset(),
        environment=runtime.environment_receipt(),code_hashes=code_hashes(),backend_policy=policy,
        adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=False,heldout_used_for_selection=False)
    rows = {};started = time.time();identities = v6.window_identity(windows)
    # The complete actual-byte disposition inventory is persisted before the first GPU forward.
    write_new_json(args.out/'intervention_inventory.json',dict(complete=True,input_binding=binding,
        protocol_sha256=PROTOCOL_SHA,intervention_inventory=inventory,arms=[row[0] for row in arm_specs(inventory)],code_hashes=code_hashes()))
    for arm,layer,group,kind in arm_specs(inventory):
        print('[group-calibration-v9 arm] '+arm,flush=True);frozen.check();need(v6.no_adapter_hooks(model),'Unexpected adapter hooks')
        table = table_for_arm(original['tables'],parent['permutations'],layer,group,kind);digest = tensor_sha(table)
        path = args.out/('calibration_'+arm+'.json');spec = execution_spec(arm)
        ac = dict(common,arm=arm,candidate_table_sha256=digest,group_replacement=dict(layer=layer,group=group,kind=kind))
        try:row = v6.evaluate(model,table,spec,windows,path,ac)
        except v6.CandidateInvalid as error:
            row = read_json(path);row.update(complete=False,error=repr(error),error_type='CandidateInvalid')
            if layer is None:row['fatal_failure']=True;v6.save_json(path,row);raise
            row['excluded_from_selection']=True
        except Exception as error:
            row = read_json(path) if path.exists() else dict(ac,**spec)
            row.update(complete=False,error=repr(error),error_type=type(error).__name__,fatal_failure=True);v6.save_json(path,row);raise
        row['frozen_source']=frozen.check();row['backend_policy_check']=v6.check_replay_backend(policy)
        need(tensor_sha(table)==digest and v6.no_adapter_hooks(model),'Table or adapter hooks changed');row['candidate_table_unchanged']=True
        if row['complete']:v6.check_ppl(row,identities);need(v6.valid_candidate(row),'Completed arm failed integrity')
        v6.save_json(path,row);rows[arm]=row
    restoration=v6.check_restoration(rows['baseline'],rows['restored_baseline']);v6.save_json(args.out/'calibration_restoration.json',restoration)
    ranking=rank_groups(rows,inventory)
    comp=dict(format=COMPARE,complete=True,stage='calibration',protocol_sha256=PROTOCOL_SHA,input_binding=binding,
        code_hashes=code_hashes(),backend_policy=policy,intervention_inventory=inventory,ranking=ranking,restoration=restoration,
        intervention_inventory_sha256=data.sha_file(args.out/'intervention_inventory.json'),
        report_sha256={arm:data.sha_file(args.out/('calibration_'+arm+'.json')) for arm in rows},elapsed_seconds=time.time()-started,
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,mk_used=False)
    v6.save_json(args.out/'calibration_comparison.json',comp)
    tables,specs,proposals=derive_candidates(original['tables'],parent['permutations'],ranking)
    metadata=candidate_metadata(binding,inventory,ranking,args.out/'calibration_comparison.json',tables,specs,proposals)
    path=args.out/'candidates.pt';write_payload(path,dict(format=CANDIDATES,**metadata,tables=tables))
    write_new_json(path.with_suffix('.json'),dict(format=CANDIDATES,**metadata,complete=True,file=path.name,
        sha256=data.sha_file(path),bytes=path.stat().st_size,code_hashes=code_hashes()))
    load_group_candidates(path,args)
    print(json.dumps(dict(complete=True,improving_groups=ranking['improving_groups'],candidate_order=list(tables),
        candidates_sha256=data.sha_file(path)),indent=2),flush=True)


if __name__=='__main__':main()
