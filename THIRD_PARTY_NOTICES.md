# Third-party notices for Mamb2_8B_FP4_Recall_SQ3.25

This notice describes the complete FP4 G16 + ridge SQ3.25 + Resurface bundle.
File paths below are relative to the complete bundle root. The Hugging Face
snapshot preserves this root under `release/`; the GitHub assets assemble
into `mamba2-8b-fp4g16-sq325-resurface-v1/`.

## License scope

| Component | License | Included license text |
|---|---|---|
| Modified NVIDIA base weights, FP4 G16 codes/scales and retained FP16 tensors | Apache-2.0 | `reference/w4/WEIGHTS_LICENSE.txt` |
| StateQuant reference kernel | Apache-2.0 | `reference/statequant/LICENSE` |
| Mamba arithmetic and separately installed `mamba-ssm` implementation | Apache-2.0 | `reference/statequant/LICENSE` contains the Apache-2.0 text; retain the installed package's own notices |
| Inherited Recall implementation, new project runtime/training/evaluation code, static state configuration and new Resurface adapter | GPL-3.0 | `LICENSE` |
| WikiText/Wikipedia text contained in any retained tokenized excerpts | Upstream dataset and Wikipedia text terms | See the dataset attribution below |

The root GPL-3.0 license does not relicense the Apache-2.0 base weights,
the Apache-2.0 StateQuant reference, or underlying dataset text. Conversely,
the base weight license does not grant an Apache-2.0 license to the project
runtime or this newly trained adapter. Prior family releases may have different
artifact-specific grants; those grants identify other checkpoints and adapters
and do not define the license of this bundle.

Redistribution should preserve this notice, applicable license texts, source
attributions, the modification notices below, and the manifests identifying
the artifacts. GPL-covered source is included in the bundle and the tagged
GitHub repository.

## NVIDIA Mamba-2 8B base and tokenizer

Original model: [`nvidia/mamba2-8b-3t-4k`, revision
`b915550c63ba9359f88f44d1f6a600d85af27302`](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302).
The [pinned original model
card](https://huggingface.co/nvidia/mamba2-8b-3t-4k/blob/b915550c63ba9359f88f44d1f6a600d85af27302/README.md)
declares Apache-2.0 and identifies this checkpoint as the pure Mamba-2 model.
The full Apache-2.0 text is included in `reference/w4/WEIGHTS_LICENSE.txt`.
The pinned upstream repository contains no separate NOTICE file.

Model authors: Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick,
Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh,
Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh, Jared Casper, Jan Kautz,
Mohammad Shoeybi, and Bryan Catanzaro. See [*An Empirical Study of Mamba-based
Language Models* (2024)](https://arxiv.org/abs/2406.07887).

**Modification notice:** EndlessChasing independently maps the original
Megatron checkpoint to the included native model configuration, casts source
BF16 tensors to the FP16 quality reference, and quantizes 114 large matrices
to E2M1 FP4 with 16-element groups, E4M3FN block scales, FP32 matrix scales,
and a fixed weight reconstruction-error range search. The remaining 393
small tensors remain FP16. The modified base files are included in `weights/`
under Apache-2.0, retaining the original attributions. No Quamba2 quantized
checkpoint is used. `weights/conversion_receipt.json` and
`weights/weight_manifest.json` record the conversion and tensor identities.
The original authors do not endorse this modification.

The bundle includes the source tokenizer
`tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model`
from the same pinned source repository. Its SHA-256 identity is
`5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
The original checkpoint SHA-256 is
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.

## Mamba implementation

The execution framework uses the separately installed
[`state-spaces/mamba`](https://github.com/state-spaces/mamba) implementation,
copyright Tri Dao and Albert Gu, under Apache-2.0. The measured installation
is `mamba-ssm` version `2.3.2.post1` from commit
`e9594ce1c732d97440f0332fdc43170a2294dbfa`.
Mamba-2 is described by Dao and Gu in [*Transformers are SSMs: Generalized
Models and Efficient Algorithms Through Structured State Space Duality*
(2024)](https://arxiv.org/abs/2405.21060).

The project's S16 and quantized-state scan arithmetic follows Mamba selective
state update, with additional packing, per-group permutations, latent carry,
frozen predictors and explicit buffer ownership. These modifications are
identified by the included source and reports; they do not claim bitwise
equivalence to every upstream execution path. The separately installed Mamba,
PyTorch, Triton and other dependencies retain their own licenses and notices.

## StateQuant

`reference/statequant/selective_state_update_pairnib.py` is preserved from
the user-supplied StateQuant archive, commit
`22156f74edf428fd192948d75a0dabf3cf152d48`, copyright 2026 Kun Yue,
Apache License 2.0. Its full license is included in
`reference/statequant/LICENSE`.

**Modification notice:** The project's codec adapts the paired-nibble
INT8/INT4 layout to the native Mamba-2 sequence loop, group-specific tables,
per-layer precision allocation, two FP8 latent coefficients, and a static
group-ridge prediction of omitted carry. The static bases, scales,
permutations and predictors are frozen and shipped in `state_config.pt`.
This modified codec is measured against its bound implementation; it is
not a claim of equivalence to the general upstream kernel.

## Recall implementation and Resurface adapter

The original Recall implementation was inherited from
[`EndlessChasing/mamb2_8B_Recall`](https://github.com/EndlessChasing/mamb2_8B_Recall)
under GPL-3.0; the license text is the root `LICENSE`. New project code,
the static state configuration and the newly trained Resurface adapter in
this bundle are distributed under GPL-3.0, subject to the separate upstream
notices above. Their exact binding and file identities are in `manifest.json`
and the training/evaluation evidence.

The adapter is inspired by [*Resurface: Multi-Binding Recall Is Latent in
Mamba's State*](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State).
It uses an independently implemented gated post-D correction across heads.
Its insertion site and training binding are specific to this bundle; it is
not presented as an exact reproduction of the original insertion site or as
an interchangeable upstream adapter. The base weight modification and this
adapter do not imply endorsement by the original authors.

## WikiText and any retained tokenized excerpts

Training and PPL evidence uses WikiText by Stephen Merity, Caiming Xiong,
James Bradbury, and Richard Socher, described in
[*Pointer Sentinel Mixture Models* (2016)](https://arxiv.org/abs/1609.07843).
The underlying articles were written by Wikipedia contributors. The pinned
dataset is [`Salesforce/wikitext`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3),
configuration `wikitext-2-raw-v1`.

The complete prose corpus is not part of the model bundle. Any small tokenized
training or numerical probe excerpts retained with source or evidence preserve
input IDs for reproduction; tokenization and cropping are transformations of
dataset text, and do not change its underlying attribution or applicable terms.
The model code license does not relicense those excerpts.

The [pinned dataset
card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
lists [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and
[GFDL](https://www.gnu.org/licenses/fdl-1.3.html) in its metadata, while its
Licensing Information section links to
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
This notice retains that upstream attribution and discrepancy. Consult the
pinned source and linked Wikipedia contributor history for applicable text
terms. Synthetic numeric recall data, token hashes and model statistics are
separate from the underlying prose text.
