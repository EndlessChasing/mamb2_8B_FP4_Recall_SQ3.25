#!/usr/bin/env python3
"""Recreate the pinned WT2 test token file for the published CPU audit.

Does not load weights, initialize CUDA, or recompute model logits. The output
is local audit input and is deliberately excluded from the public reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_SHA = "5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09"
TEXT_SHA = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
TOKEN_SHA = "5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608"
TOKENIZER_NAME = "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    need(args.out.name == "tokens.int64le" and not args.out.exists() and
         not args.out.is_symlink() and not args.out.resolve().is_relative_to(bundle),
         "Fresh tokens.int64le outside the sealed bundle required")
    tokenizer = bundle / "tokenizer" / TOKENIZER_NAME
    need(hashlib.sha256(tokenizer.read_bytes()).hexdigest() == TOKENIZER_SHA,
         "Published tokenizer hash differs")

    from datasets import load_dataset
    import sentencepiece as spm
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                           split="test", revision=REVISION)
    text = "\n\n".join(dataset["text"])
    need(hashlib.sha256(text.encode()).hexdigest() == TEXT_SHA,
         "Pinned official test text differs")
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    ids = processor.encode_as_ids(text)
    need(len(ids) == 300964 and all(0 <= value < 256000 for value in ids),
         "Pinned complete token population differs")
    raw = b"".join(struct.pack("<q", value) for value in ids)
    need(hashlib.sha256(raw).hexdigest() == TOKEN_SHA, "Pinned test token hash differs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(complete=True, tokens=len(ids), target_tokens=len(ids)-1,
                          bytes=len(raw), sha256=TOKEN_SHA, cuda_initialized=False,
                          model_logits_recomputed=False), indent=2))


if __name__ == "__main__":
    main()
