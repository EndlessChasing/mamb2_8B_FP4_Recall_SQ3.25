"""Frozen prose PPL and a public synthetic multi-key recall protocol.

This MK protocol is new and is not interchangeable with private Resurface scores.
Use --execution tokenwise to round the recurrent FP16 state after every token.
The faster prefill protocol is reported separately and is never called tokenwise.
The source-only experiment uses these scoring functions with an external adapter.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .calibration import WIKITEXT_REVISION, load_wikitext_tokens, window_starts
from .runtime import (SOURCE_CHECKPOINT_SHA256, SentencePieceTokenizer, backbone_tokenwise,
                      cache_bytes, load_source_model, make_cache,
                      environment_receipt, gpu_memory_receipt, sha256_file, token_digest)


def ppl_windows(ids, seqlen, nwin=None):
    if nwin is not None:
        starts = window_starts(len(ids), seqlen+1, nwin)
        return [(start, ids[start:start+seqlen+1]) for start in starts]
    return [(start, ids[start:min(start+seqlen+1, len(ids))])
            for start in range(0, len(ids)-1, seqlen)]


@torch.inference_mode()
def evaluate_ppl(model, windows, execution="prefill", logits_chunk=64):
    device = next(model.parameters()).device
    nll, count, rows = 0.0, 0, []
    started = time.time()
    state_memory = None
    for index, (start, window) in enumerate(windows):
        tokens = window.to(device)
        if execution == "tokenwise":
            hidden, cache = backbone_tokenwise(model, tokens[:-1][None], torch.float16)
            state_memory = cache_bytes(cache)
            if any(t.dtype != torch.float16 for pair in cache.key_value_memory_dict.values() for t in pair):
                raise RuntimeError("Tokenwise protocol requires FP16 state and conv cache")
            del cache
        else:
            hidden = model.backbone(tokens[:-1][None])
        if not torch.isfinite(hidden).all():
            raise RuntimeError(f"Nonfinite hidden states in PPL window {index}")
        window_nll = 0.0
        for position in range(0, hidden.shape[1], logits_chunk):
            end = min(position+logits_chunk, hidden.shape[1])
            logits = model.lm_head(hidden[:, position:end]).float()
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Nonfinite logits in PPL window {index}")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   tokens[position+1:end+1], reduction="sum")
            window_nll += float(loss)
            if not math.isfinite(window_nll):
                raise RuntimeError(f"Nonfinite loss in PPL window {index}")
            del logits, loss
        targets = len(window)-1
        nll += window_nll
        count += targets
        rows.append({"start": start, "input_tokens": targets, "target_tokens": targets,
                     "token_sha256_int64le": token_digest(window.numpy()),
                     "nll": window_nll, "ppl": math.exp(window_nll/targets)})
        del hidden, tokens
        if (index+1) % 4 == 0 or index == 0:
            print(f"[PPL {execution}] {index+1}/{len(windows)}, ppl={math.exp(nll/count):.6f}, "
                  f"{time.time()-started:.1f}s", flush=True)
    return {"ppl": math.exp(nll/count), "nll": nll, "target_tokens": count,
            "execution": execution, "windows": rows, "elapsed_seconds": time.time()-started,
            "cache_dtype": "float16 each token" if execution == "tokenwise" else "SSD scan internal precision",
            "cache_bytes": state_memory, "logits_chunk_tokens": logits_chunk,
            "score": "next-token cross entropy; no BOS/EOS added; each window starts from zero state"}


def synthetic_mk_cases(split="test", samples_per_cell=8, sizes=(16, 64)):
    """Frozen public prompts; dev and test use disjoint keys and value ranges."""
    if split not in ("validation", "test"):
        raise ValueError(split)
    offset = 0 if split == "validation" else 100000
    rng = random.Random(834901 if split == "validation" else 834902)
    cases = []
    for size in sizes:
        for template in range(3):
            for sample in range(samples_per_cell):
                keys = rng.sample(range(100000+offset, 180000+offset), size)
                values = rng.sample(range(600000+offset, 680000+offset), size)
                pairs = list(zip(keys, values))
                # Query positions cycle across the sequence, including both ends.
                target_position = (sample * max(1, size-1)) // max(1, samples_per_cell-1)
                key, value = pairs[target_position]

                def render(records):
                    if template == 0:
                        body = "\n".join(f"{k}: {v}" for k, v in records)
                        return f"Key-value records:\n{body}\n\nLookup the value for key {key}.\nValue:"
                    if template == 1:
                        body = "\n".join(f"Key {k} has value {v}." for k, v in records)
                        return f"{body}\n\nQuestion: What is the value of key {key}?\nAnswer:"
                    body = "\n".join(f"{k} -> {v}" for k, v in records)
                    return f"Dictionary:\n{body}\n\nReturn the stored value.\n{key} ->"

                case_id = f"{split}-n{size}-t{template}-s{sample}"
                cases.append({"id": case_id, "N": size, "template": template,
                              "query_position": target_position, "key": key, "answer": str(value),
                              "prompt": render(pairs), "condition": "normal"})
                # Controls replace the target with an unrelated held-out key/value.
                # Keep record count and template fixed; the target value is absent.
                removed = pairs.copy()
                removed[target_position] = (190000+offset+sample, 690000+offset+sample)
                cases.append({"id": case_id+"-removed", "N": size, "template": template,
                              "query_position": target_position, "key": key, "answer": str(value),
                              "prompt": render(removed), "condition": "target_removed"})
    return cases


@torch.inference_mode()
def generate_greedy(model, tokenizer, prompt, max_new_tokens=12, execution="prefill"):
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None]
    if ids.shape[1] == 0:
        raise ValueError("Empty recall prompt")
    cache = make_cache(model, ids.shape[1]+max_new_tokens, state_dtype=torch.float16)
    if execution == "tokenwise":
        for position in range(ids.shape[1]):
            cache.seqlen_offset = position
            hidden = model.backbone(ids[:, position:position+1], inference_params=cache)
    else:
        hidden = model.backbone(ids, inference_params=cache)
        hidden = hidden[:, -1:]
    generated = []
    for position in range(max_new_tokens):
        logits = model.lm_head(hidden[:, -1:])
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite generation logits")
        token = int(logits.argmax(-1).item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        cache.seqlen_offset = ids.shape[1]+position
        hidden = model.backbone(torch.tensor([[token]], device=device), inference_params=cache)
    return tokenizer.decode(generated), generated, ids.shape[1], cache_bytes(cache)


def evaluate_mk(model, tokenizer, split="test", samples_per_cell=8, execution="prefill"):
    cases = synthetic_mk_cases(split, samples_per_cell)
    rows = []
    started = time.time()
    for index, case in enumerate(cases):
        output, generated, count, memory = generate_greedy(model, tokenizer, case["prompt"], execution=execution)
        first = re.search(r"(?<!\d)\d{6}(?!\d)", output)
        prediction = first.group() if first else None
        ids = tokenizer.encode(case["prompt"])
        rows.append({**{k: v for k, v in case.items() if k != "prompt"},
                     "prompt_tokens": count, "prompt_token_sha256_int64le": token_digest(ids),
                     "prompt": case["prompt"], "output": output, "generated_ids": generated,
                     "prediction": prediction, "correct": prediction == case["answer"],
                     "cache_bytes": memory})
        if (index+1) % 8 == 0 or index == 0:
            print(f"[public MK {execution}] {index+1}/{len(cases)}, {time.time()-started:.1f}s", flush=True)
    summary = {}
    for condition in ("normal", "target_removed"):
        selected = [row for row in rows if row["condition"] == condition]
        correct = sum(row["correct"] for row in selected)
        summary[condition] = {"correct": correct, "count": len(selected), "accuracy": correct/len(selected)}
    cells = []
    for size in (16, 64):
        for template in range(3):
            for condition in ("normal", "target_removed"):
                selected = [row for row in rows if row["N"] == size and row["template"] == template
                            and row["condition"] == condition]
                cells.append({"N": size, "template": template, "condition": condition,
                              "correct": sum(row["correct"] for row in selected), "count": len(selected),
                              "query_positions": [row["query_position"] for row in selected],
                              "paired_case_ids": [row["id"] for row in selected]})
    return {"protocol": "public-mamba8-mk-v1", "split": split,
            "execution": execution, "cache_dtype": "float16",
            "prefill_rounding": "each token" if execution == "tokenwise" else "at end of prefill",
            "generation": "greedy, at most 12 tokens, stop at EOS",
            "matching": "first standalone six-digit integer equals answer",
            "private_resurface_comparable": False, "summary": summary, "cells": cells, "rows": rows,
            "interpretation": "Small engineering screen; not statistical equivalence. Near-zero baseline recall cannot establish retained recall capacity.",
            "elapsed_seconds": time.time()-started}

