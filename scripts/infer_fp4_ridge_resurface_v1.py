#!/usr/bin/env python3
"""Generate with the complete public FP4 G16/SQ3.25 ridge Resurface bundle.

Uses real packed weights from the bundle, decoded to resident FP16, and the
exact original PredictorState codec. No original NVIDIA weights are required.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_fp4_ridge_resurface_v1 import verify_bundle, need, STATE_FORMAT


def read_state(bundle, manifest):
    import torch
    from mamba2_recall import resurface_native as native
    state = torch.load(bundle / 'state_config.pt', map_location='cpu', weights_only=True)
    need(state['format'] == STATE_FORMAT and state['provenance'] == manifest['state_provenance'],
         'State export/provenance differs')
    provenance = state['provenance']
    table = state['table']
    need(table.dtype == torch.uint8 and tuple(table.shape) == (56, 8, 128)
         and native.tensor_hash(table) == provenance['table_sha256']
         and torch.equal(table.sort(-1).values, torch.arange(128, dtype=torch.uint8).expand_as(table)),
         'Final state permutation table differs')
    bases, scales, predictors = (state[key] for key in ('bases', 'scales', 'stackedpredictors'))
    need(state['layouts'] == provenance['layouts'] and len(bases) == len(scales) == len(predictors) == 56,
         'Static state inventory/layouts differ')
    fingerprints = dict(bases=[native.tensor_hash(value) for value in bases],
                        scales=[native.tensor_hash(value) for value in scales],
                        predictors=[[native.tensor_hash(value) for value in layer] for layer in predictors])
    need(fingerprints == provenance['static_tensor_sha256'], 'Static state tensor hashes differ')
    need(all(value.dtype == torch.float16 and value.is_contiguous() and bool(torch.isfinite(value).all())
             for collection in (bases, scales, predictors) for value in collection),
         'Static state dtype/finite/contiguity differs')
    total = table.numel() + sum(value.numel() * value.element_size()
                                for collection in (bases, scales, predictors) for value in collection)
    need(total == manifest['state_config_payload_bytes'] == 3842304, 'Static state tensor payload differs')
    return table, tuple(state['layouts']), bases, scales, predictors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prompt')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--verify-only', action='store_true', help='Full file verification without torch/CUDA')
    parser.add_argument('--without-adapter', action='store_true', help='Use the frozen unadapted 8.40828 ridge state configuration')
    parser.add_argument('--smoke-check', action='store_true', help='Exactly replay one published MK generation case')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    manifest = verify_bundle(args.bundle)
    if args.verify_only:
        print(json.dumps({'verified': True, 'candidate_ppl': manifest['candidate_ppl'],
                          'scope': manifest['scope']}, indent=2))
        return
    if args.max_new_tokens <= 0 or (not args.prompt and not args.smoke_check):
        parser.error('Generation requires a prompt or --smoke-check and positive max-new-tokens')
    if args.smoke_check and args.without_adapter:
        parser.error('--smoke-check validates the released adapted generation')

    import torch
    from mamba2_recall import runtime, resurface_native as native
    from mamba2_recall.fp4_checkpoint import load_packed_model
    from fp4_zero_predictor_codec_v1 import PredictorState
    from evaluate_resurface_more import pin_replay_backend, check_replay_backend
    need(torch.cuda.is_available(), 'Recorded CUDA/Mamba/Triton stack is required')
    torch.set_num_threads(8)
    torch.manual_seed(20260929)
    torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    backend = pin_replay_backend()
    need(backend == manifest['backend_policy'], 'Execution stack/pinned backend differs from measured release')
    tokenizer = runtime.SentencePieceTokenizer(args.bundle / 'tokenizer')
    expected_case = None
    if args.smoke_check:
        report = json.loads((args.bundle / 'evidence/ridge_resurface.json').read_text())
        expected_case = next(case for case in report['mk']['rows'] if case['condition'] == 'normal')
        args.prompt = expected_case['prompt']
        args.max_new_tokens = len(expected_case['generated_ids'])
    prompt_ids = tokenizer.encode(args.prompt)
    need(0 < len(prompt_ids) <= 4096, 'Prompt must contain 1 to 4096 tokens')
    state = read_state(args.bundle, manifest)
    model = load_packed_model(args.bundle / 'weights', device='cuda')
    ledger = json.loads((args.bundle / 'weights/conversion_receipt.json').read_text())
    expected_hashes = {name: item['decoded_sha256'] for name, item in ledger['tensors'].items()}
    generated = []
    manager = (contextlib.nullcontext() if args.without_adapter else
               native.install_fp16(model, args.bundle / 'adapter_fp16.pt',
                                   expected_binding=manifest['adapter_binding'], expected_base_hashes=expected_hashes))
    with torch.inference_mode(), manager as bank, PredictorState(model, *state) as execution:
        hidden = execution.backbone(torch.tensor([prompt_ids], device='cuda'), reset=True)[:, -1:]
        for step in range(args.max_new_tokens):
            logits = model.lm_head(hidden)
            need(bool(torch.isfinite(logits).all()), 'Nonfinite generation logits')
            token = int(logits.argmax(-1).item())
            generated.append(token)
            if token == tokenizer.eos_token_id or step + 1 == args.max_new_tokens:
                break
            hidden = execution.backbone(torch.tensor([[token]], device='cuda'), reset=False)
        execution.assert_finite_cache()
        descriptor = execution.storage_descriptor()
        persistent = descriptor['total_cache_bytes'] + sum(descriptor[key] for key in
                         ('static_basis_bytes', 'static_scale_bytes', 'static_predictor_bytes'))
        adapter_bytes = 0 if bank is None else sum(value.untyped_storage().nbytes() for value in bank.masters.values())
        need(persistent == manifest['persistent_state_side_bytes'] == 32284928,
             'Actual complete state storage differs')
        need(adapter_bytes == (0 if args.without_adapter else manifest['adapter_fp16_payload_bytes']),
             'Actual adapter storage differs')
        check_replay_backend(backend)
    if expected_case is not None:
        need(generated == expected_case['generated_ids'], 'Public runtime MK generation does not exactly replay')
    result = dict(prompt=args.prompt, completion=tokenizer.decode(generated), generated_token_ids=generated,
                  prompt_tokens=len(prompt_ids), adapter_active=not args.without_adapter,
                  smoke_case_id=None if expected_case is None else expected_case['id'],
                  published_mk_generation_exact=expected_case is not None,
                  persistent_state_side_bytes=persistent, adapter_storage_bytes=adapter_bytes,
                  state_plus_adapter_bytes=persistent + adapter_bytes,
                  packed_weight_payload_bytes=manifest['packed_weight_payload_bytes'],
                  decoded_resident_weight_bytes=manifest['decoded_resident_weight_bytes'],
                  packed_resident_kernel=False,
                  memory_scope='Persistent tensor payload; activations, scratch, CPU copies and allocator reserve excluded')
    if args.out:
        need(not args.out.exists(), 'Fresh inference output receipt required')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
