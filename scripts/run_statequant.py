#!/usr/bin/env python3
"""Run frozen TRAIN calibration and paired state-cache PPL/MK experiment."""
import argparse
import gc
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime, resurface_data as data
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from mamba2_recall.resurface_native import install_fp16, tensor_hash
from mamba2_recall.state_quant import StateQuant

ADAPTER_SHA = 'e8b2b4dfe69f8e85dc9e147c9aeaa3297cff14f4043e558bb1795ad476c1fca0'
TRAIN_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
PILOT_WINDOWS = [0, 32, 64, 96]
PILOT_SAMPLES = [0, 9, 18, 27, 36, 45, 54, 63]


def save_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def code_hashes():
    paths = sorted((ROOT/'mamba2_recall').glob('*.py')) + sorted((ROOT/'scripts').glob('*.py'))
    paths += [ROOT/'docs'/'PROTOCOL.md']
    return {str(path.relative_to(ROOT)): runtime.sha256_file(path) for path in paths}


@torch.inference_mode()
def probe(model, ids):
    ids = ids[:32].cuda()[None]
    native, cache = runtime.backbone_tokenwise(model, ids)
    del cache
    with StateQuant(model, 's16') as execution:
        pieces = [execution.backbone(ids[:, i:i+1]) for i in range(ids.shape[1])]
        packed = torch.cat(pieces, 1)
        breakdown = execution.cache_breakdown()
    diff = native.float() - packed.float()
    relative_rms = float(diff.square().mean().sqrt() / native.float().square().mean().sqrt())
    native_top = model.lm_head(native[:, -8:]).argmax(-1)
    packed_top = model.lm_head(packed[:, -8:]).argmax(-1)
    top_equal = int((native_top == packed_top).sum())
    restored, cache = runtime.backbone_tokenwise(model, ids)
    restoration_equal = torch.equal(native, restored)
    del cache
    result = {'tokens': ids.shape[1], 'token_sha256': runtime.token_digest(ids.cpu().numpy()),
              'native_vs_s16_max_abs': float(diff.abs().max()),
              'native_vs_s16_relative_rms': relative_rms,
              'last_8_argmax_matches': top_equal, 'native_restored_bitwise': restoration_equal,
              's16_cache': breakdown,
              'scope': '32-token native recurrent versus wrapper recurrent; both include frozen Recall adapter'}
    if not restoration_equal or not math.isfinite(relative_rms) or relative_rms > 0.01:
        raise RuntimeError(f'Native runtime control failed: {result}')
    return result


@torch.inference_mode()
def calibrate(model, train, out, protocol_sha):
    started = time.time()
    with StateQuant(model, 's16', collect_stats=True) as execution:
        for index in range(8):
            hidden = execution.backbone(train[index, :512].cuda()[None], reset=True)
            if not bool(torch.isfinite(hidden).all()):
                raise RuntimeError('Nonfinite calibration hidden states')
            del hidden
            print(f'[calibration] {index+1}/8, {time.time()-started:.1f}s', flush=True)
        stats = execution.statistics()
        if not bool(torch.isfinite(stats['mean_abs']).all()):
            raise RuntimeError('Nonfinite calibration statistics')
        permutation = torch.argsort(stats['mean_abs'], dim=-1, descending=True, stable=True).to(torch.uint8)
        workspace = execution.cache_breakdown()
    expected_count = 8*512*16*64
    if not bool((stats['sample_count_per_group'] == expected_count).all()):
        raise RuntimeError('Calibration counts differ from frozen protocol')
    payload = {'format': 'MAMBA2_RECALL_STATEQUANT_CALIBRATION_V1',
               'protocol_sha256': protocol_sha, 'adapter_sha256': ADAPTER_SHA,
               'source_sha256': runtime.SOURCE_CHECKPOINT_SHA256,
               'train_file_sha256': TRAIN_SHA, 'permutations': permutation,
               'statistics': stats, 'selection': 'first 8 rows, first 512 tokens each'}
    path = out/'calibration.pt'
    torch.save(payload, path)
    restored = torch.load(path, map_location='cpu', weights_only=True)
    if not torch.equal(restored['permutations'], permutation):
        raise RuntimeError('Permutation serialization failed')
    ordered = torch.gather(stats['mean_abs'], -1, permutation.long())
    receipt = {'file': path.name, 'sha256': runtime.sha256_file(path), 'bytes': path.stat().st_size,
               'protocol_sha256': protocol_sha, 'train_file_sha256': TRAIN_SHA,
               'calibration_tokens': 4096, 'heldout_used': False,
               'token_hashes': [runtime.token_digest(train[i, :512].numpy()) for i in range(8)],
               'sample_count_per_group': stats['sample_count_per_group'].tolist(),
               'permutation_payload_bytes': permutation.numel(),
               'dead_coordinate_abs_mass_fraction': float(ordered[..., 80:].sum()/ordered.sum()),
               'statistics_semantics': stats['semantics'], 'cache_and_workspace': workspace,
               'elapsed_seconds': time.time()-started}
    save_json(out/'calibration.json', receipt)
    return permutation, receipt


@torch.inference_mode()
def evaluate(model, tokenizer, mode, permutations, windows, cases, path, common):
    result = {**common, 'mode': mode, 'complete': False, 'ppl': {'windows': []},
              'mk': {'rows': []}}
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    with StateQuant(model, mode, permutations if mode == 'sq3p25' else None) as execution:
        nll, count = 0.0, 0
        for index, (start, window) in enumerate(windows):
            tokens = window.cuda()
            hidden = execution.backbone(tokens[:-1][None], reset=True)
            if not bool(torch.isfinite(hidden).all()):
                raise RuntimeError('Nonfinite PPL hidden states')
            loss_sum = 0.0
            for pos in range(0, hidden.shape[1], 64):
                end = min(pos+64, hidden.shape[1])
                logits = model.lm_head(hidden[:, pos:end]).float()
                loss = F.cross_entropy(logits.reshape(-1, 256000), tokens[pos+1:end+1], reduction='sum')
                loss_sum += float(loss)
                del logits, loss
            if not math.isfinite(loss_sum):
                raise RuntimeError('Nonfinite PPL loss')
            n = len(window)-1
            nll += loss_sum
            count += n
            result['ppl']['windows'].append({'start': start, 'target_tokens': n,
                'token_sha256_int64le': runtime.token_digest(window.numpy()),
                'nll': loss_sum, 'ppl': math.exp(loss_sum/n)})
            result['ppl'].update(nll=nll, target_tokens=count, ppl=math.exp(nll/count))
            result['cache'] = execution.cache_breakdown()
            save_json(path, result)
            print(f'[{mode} PPL] {index+1}/{len(windows)} ppl={math.exp(nll/count):.6f}, {time.time()-started:.1f}s', flush=True)
            del hidden, tokens
        result['ppl']['elapsed_seconds'] = time.time()-started
        mk_start = time.time()
        for index, case in enumerate(cases):
            encoded = tokenizer.encode(case['prompt'])
            ids = torch.tensor(encoded, device='cuda', dtype=torch.long)[None]
            hidden = execution.backbone(ids, reset=True)[:, -1:]
            generated = []
            for step in range(12):
                logits = model.lm_head(hidden)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError('Nonfinite MK logits')
                token = int(logits.argmax(-1).item())
                generated.append(token)
                if token == tokenizer.eos_token_id or step == 11:
                    break
                hidden = execution.backbone(torch.tensor([[token]], device='cuda'))
            output = tokenizer.decode(generated)
            match = re.search(r'(?<!\d)\d{6}(?!\d)', output)
            prediction = match.group() if match else None
            result['mk']['rows'].append({**case, 'prompt_tokens': len(encoded),
                'prompt_token_sha256_int64le': runtime.token_digest(encoded),
                'generated_ids': generated, 'output': output, 'prediction': prediction,
                'correct': prediction == case['answer']})
            if index == 0 or (index+1) % 8 == 0:
                save_json(path, result)
                print(f'[{mode} MK] {index+1}/{len(cases)}, {time.time()-mk_start:.1f}s', flush=True)
        result['mk']['summary'] = {}
        for condition in ('normal', 'target_removed'):
            rows = [r for r in result['mk']['rows'] if r['condition'] == condition]
            correct = sum(r['correct'] for r in rows)
            result['mk']['summary'][condition] = {'correct': correct, 'count': len(rows), 'accuracy': correct/len(rows)}
        result['mk']['elapsed_seconds'] = time.time()-mk_start
        result['gpu_memory'] = runtime.gpu_memory_receipt()
    result.update(complete=True, elapsed_seconds=time.time()-started)
    save_json(path, result)
    return result


def comparison(control, candidate, stage):
    if not control['complete'] or not candidate['complete']:
        raise RuntimeError('Cannot compare incomplete reports')
    p0, p1 = control['ppl']['ppl'], candidate['ppl']['ppl']
    mk0, mk1 = control['mk']['summary']['normal'], candidate['mk']['summary']['normal']
    delta = mk1['accuracy']-mk0['accuracy']
    stop = (p1/p0-1 > 0.05 or delta < -0.10)
    result = {'ppl_s16': p0, 'ppl_sq3p25': p1, 'ppl_relative_change': p1/p0-1,
              'mk_s16': mk0, 'mk_sq3p25': mk1, 'mk_accuracy_delta': delta,
              'pilot_stop': stop, 'stage': stage,
              'status': 'STOP: fixed candidate failed pilot' if stop else 'Pilot survived; full validation required'}
    if stage == 'full':
        import numpy as np
        a = {r['id']: r for r in control['mk']['rows'] if r['condition'] == 'normal'}
        b = {r['id']: r for r in candidate['mk']['rows'] if r['condition'] == 'normal'}
        if set(a) != set(b):
            raise RuntimeError('MK pairs differ')
        differences = np.asarray([int(b[k]['correct'])-int(a[k]['correct']) for k in sorted(a)])
        rng = np.random.default_rng(20260928)
        boot = differences[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
        lower, upper = np.quantile(boot, [0.025, 0.975])
        passed = p1/p0-1 <= 0.01 and lower >= -0.02
        result.update(mk_paired_bootstrap_95ci=[float(lower), float(upper)],
                      full_gate_pass=bool(passed), status='PASS' if passed else 'Full gate not passed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--train-tokens', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts'/'statequant_v1')
    parser.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/f'{args.stage}_comparison.json').exists():
        raise FileExistsError('A completed experiment exists; preserve it and choose a new output directory')
    protected = ('calibration.pt', 'calibration.json', 'pilot_s16.json', 'pilot_sq3p25.json')
    if args.stage == 'pilot' and any((args.out/name).exists() for name in protected):
        raise FileExistsError('Pilot artifacts already exist; preserve them and choose a fresh output directory')
    if args.stage == 'full' and any((args.out/f'full_{m}.json').exists() for m in ('s16', 'sq3p25')):
        raise FileExistsError('Full reports already exist; preserve them before a new independent run')
    full_calibration = None
    if args.stage == 'full':
        pilot = json.loads((args.out/'pilot_comparison.json').read_text())
        if pilot['pilot_stop'] or pilot['stage'] != 'pilot':
            raise RuntimeError('Full validation requires a surviving frozen pilot')
        pilot_reports = {}
        for mode, digest in pilot['report_sha256'].items():
            if runtime.sha256_file(args.out/f'pilot_{mode}.json') != digest:
                raise RuntimeError('Pilot report changed')
            pilot_reports[mode] = json.loads((args.out/f'pilot_{mode}.json').read_text())
        if set(pilot_reports) != {'s16', 'sq3p25'}:
            raise RuntimeError('Both completed pilot arms are required')
        full_calibration = json.loads((args.out/'calibration.json').read_text())
        if runtime.sha256_file(args.out/'calibration.pt') != full_calibration['sha256']:
            raise RuntimeError('Frozen calibration file changed')
        current_hashes = code_hashes()
        for report in pilot_reports.values():
            if not report['complete'] or report['calibration'] != full_calibration:
                raise RuntimeError('Calibration no longer matches the scored pilot candidate')
            for relative, digest in report['code_hashes'].items():
                if relative.startswith('mamba2_recall/') and current_hashes.get(relative) != digest:
                    raise RuntimeError(f'Core execution changed after pilot: {relative}')
    torch.set_num_threads(8)
    torch.manual_seed(20260928)
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ROOT/'pretrained'/'adapter_fp16.pt'
    if runtime.sha256_file(adapter) != ADAPTER_SHA or runtime.sha256_file(args.train_tokens) != TRAIN_SHA:
        raise RuntimeError('Pinned adapter or TRAIN tensor differs')
    train = torch.load(args.train_tokens, map_location='cpu', weights_only=True)
    if train.dtype != torch.long or tuple(train.shape) != (448, 2048):
        raise RuntimeError('TRAIN tensor geometry differs')
    protocol_sha = runtime.sha256_file(ROOT/'docs'/'PROTOCOL.md')
    if full_calibration is not None and (pilot['protocol_sha256'] != protocol_sha
            or full_calibration['protocol_sha256'] != protocol_sha):
        raise RuntimeError('Protocol changed after pilot')
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    model = runtime.load_source_model(args.source_dir)
    with install_fp16(model, adapter) as bank:
        adapter_hashes = {k: tensor_hash(v) for k, v in bank.masters.items()}
        control_probe = probe(model, train[0])
        save_json(args.out/f'{args.stage}_native_control.json', control_probe)
        if full_calibration is None:
            permutations, calibration = calibrate(model, train, args.out, protocol_sha)
        else:
            payload = torch.load(args.out/'calibration.pt', map_location='cpu', weights_only=True)
            if (payload['protocol_sha256'] != protocol_sha or payload['adapter_sha256'] != ADAPTER_SHA
                    or payload['source_sha256'] != runtime.SOURCE_CHECKPOINT_SHA256
                    or payload['train_file_sha256'] != TRAIN_SHA):
                raise RuntimeError('Frozen calibration binding differs')
            permutations, calibration = payload['permutations'], full_calibration
        del train
        gc.collect()
        ids, dataset = load_wikitext_tokens(tokenizer, 'validation')
        windows = ppl_windows(ids, 2048)
        if len(windows) != 130 or sum(len(w)-1 for _, w in windows) != 264764:
            raise RuntimeError('Pinned validation tokenization differs')
        if args.stage == 'pilot':
            windows = [windows[i] for i in PILOT_WINDOWS]
            cases = [r for r in data.frozen_development_cases() if r['sample'] in PILOT_SAMPLES]
            if len(cases) != 96:
                raise RuntimeError('Pilot cases differ')
        else:
            cases = data.generate_cases('confirm')
        common = {'format': 'MAMBA2_RECALL_STATEQUANT_EVAL_V1', 'stage': args.stage,
                  'protocol_sha256': protocol_sha, 'code_hashes': code_hashes(),
                  'source_checkpoint_sha256': runtime.SOURCE_CHECKPOINT_SHA256,
                  'adapter_sha256': ADAPTER_SHA, 'calibration': calibration,
                  'dataset': dataset, 'environment': runtime.environment_receipt(),
                  'execution': 'serial recurrence with per-token compressed carry; native prompt conv and projections',
                  'native_control': control_probe}
        results = {}
        for mode in ('s16', 'sq3p25'):
            results[mode] = evaluate(model, tokenizer, mode, permutations, windows, cases,
                                     args.out/f'{args.stage}_{mode}.json', common)
            frozen = bank.assert_base_frozen()
            if adapter_hashes != {k: tensor_hash(v) for k, v in bank.masters.items()}:
                raise RuntimeError('Frozen adapter values changed')
            results[mode]['frozen_parameters'] = {**frozen, 'adapter_content_unchanged': True}
            save_json(args.out/f'{args.stage}_{mode}.json', results[mode])
        outcome = comparison(results['s16'], results['sq3p25'], args.stage)
        outcome['protocol_sha256'] = protocol_sha
        outcome['report_sha256'] = {m: runtime.sha256_file(args.out/f'{args.stage}_{m}.json') for m in results}
        save_json(args.out/f'{args.stage}_comparison.json', outcome)
        print(json.dumps(outcome, indent=2), flush=True)


if __name__ == '__main__':
    main()
