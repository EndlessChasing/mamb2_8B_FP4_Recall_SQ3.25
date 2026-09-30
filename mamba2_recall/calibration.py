"""Pinned WikiText tokenization and window selection for recall controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .runtime import (MODEL_CONFIG, SOURCE_CHECKPOINT_SHA256, SentencePieceTokenizer,
                      environment_receipt, gpu_memory_receipt, load_source_model, sha256_file, token_digest)

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def window_starts(token_count, seqlen, nwin, offset=0):
    """Evenly spread, nonoverlapping windows; never silently reuse short data."""
    usable = token_count - offset
    if seqlen <= 0 or nwin <= 0 or usable < seqlen * nwin:
        raise ValueError("Insufficient tokens for the requested nonoverlapping windows")
    if nwin == 1:
        return [offset]
    span = usable - seqlen
    starts = [offset + (index * span) // (nwin-1) for index in range(nwin)]
    if any(b-a < seqlen for a, b in zip(starts, starts[1:])):
        raise ValueError("Calibration windows overlap")
    return starts


def load_wikitext_tokens(tokenizer, split, revision=WIKITEXT_REVISION):
    from datasets import load_dataset
    kwargs = {"revision": revision} if revision else {}
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, **kwargs)
    text = "\n\n".join(dataset["text"])
    ids = tokenizer.encode(text)
    metadata = {
        "dataset": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
        "split": split, "revision_argument": revision, "dataset_fingerprint": dataset._fingerprint,
        "document_join": "two newline characters", "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_stream_sha256_int64le": token_digest(ids), "total_tokens": len(ids),
        "tokenizer_sha256": tokenizer.sha256, "automatic_special_tokens": False,
    }
    return torch.tensor(ids, dtype=torch.long), metadata


