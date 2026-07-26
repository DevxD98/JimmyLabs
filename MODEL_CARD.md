# Model Card: JimmyLabs

> **Status:** Living Document. This model card is updated as the model scales from v0.1 to v1.0.

## Model Details
- **Name:** JimmyLabs
- **Versions:** v0.1 (~0.82M) and v0.2 (~2.7M) have completed runs. v0.3 and v1.0 remain planned; see [`SPEC.md`](SPEC.md) §3 for the intended configurations.
- **Model Type:** Decoder-only, causal self-attention, autoregressive next-token language model.
- **License:** MIT License.
- **Hardware:** Built, trained, and tested entirely on a MacBook Air M1 (8GB unified memory) using PyTorch MPS/Metal.

## Intended Use
- **Primary Use Case:** **Educational only.** JimmyLabs is designed to be understood down to the last gradient, serving as a pedagogical tool for those learning how transformer-based language models work from scratch.
- **Out-of-Scope Uses:** **NOT for production.** This model is tiny (1–4 million parameters). It cannot perform complex reasoning, code generation, translation, or general-purpose instruction following. It will produce simple, often-wrong text. Do not deploy this in user-facing applications.

## Training Data
The data dictates what a model of this capacity can learn. We explicitly match data complexity to the model's 1–4M parameter capacity.
- **Corpora:**
  - *Shakespeare (tiny):* ~1 MB public domain dataset used for v0.1 as a sanity/overfit test.
  - *TinyStories:* A dataset of syntactically simple short stories used as the primary corpus for v0.2+.
- **Data Details & Licensing:** Refer to [`docs/17_DATASET_GUIDE.md`](docs/17_DATASET_GUIDE.md) and [`datasets/SOURCES.md`](datasets/SOURCES.md) for full provenance, versioning, cleaning pipelines, and licenses of the corpora used.

## Architecture
- **Description:** A from-scratch implementation of the standard GPT-style transformer.
- **Key Features:** Pre-norm LayerNorm, GELU activations, learned positional embeddings, and output weight tying ($W_{\text{out}} = W_{\text{token}}^T$).
- **Technical Specification:** See [`SPEC.md`](SPEC.md) for precise tensor shapes, parameter counts, and pipeline descriptions.

## Evaluation
We hold strictly to the evaluation contract defined in [`SPEC.md`](SPEC.md) §13.
- **v0.1 — Shakespeare, 0.82M parameters:** held-out validation loss **1.54**, recorded in
  [`docs/21_DEVLOG.md`](docs/21_DEVLOG.md); fixed-prompt output is preserved in
  [`outputs/trained_shakespeare_sample.txt`](outputs/trained_shakespeare_sample.txt).
  The measured baseline benchmark reports **65,027 train tokens/sec**, **103 generated
  tokens/sec**, and **257.6 MB MPS driver memory** ([`benchmarks/001_baseline.md`](benchmarks/001_baseline.md)).
- **v0.2 — TinyStories, 2.745M parameters:** best held-out validation loss **0.8607** and
  validation perplexity **≈2.37** at step 2800/3375; see
  [`benchmarks/004_v0_2.md`](benchmarks/004_v0_2.md). Fixed-prompt output is preserved in
  [`outputs/trained_tinystories_v0_2_sample.txt`](outputs/trained_tinystories_v0_2_sample.txt).
  The absolute architecture benchmark reports **21,000 train tokens/sec**, **28 generated
  tokens/sec**, and **1,599.0 MB MPS driver memory** in the same benchmark record.
- **v0.3 / v1.0:** no training or evaluation run exists yet; metrics remain unreported.
- *Note: These are separate absolute measurements on different models and corpora, not a
  controlled v0.1-v0.2 speed comparison. No unmeasured metrics are inferred.*

## Ethical Considerations
- **Hallucinations & Reliability:** As a very small model trained on narrow corpora, JimmyLabs will hallucinate confidently. It does not possess real-world knowledge or factual accuracy.
- **Safety:** The model has not been subjected to RLHF or safety fine-tuning. It may reflect biases present in its training data (e.g., historical biases in Shakespeare). It must not be relied upon for safety-critical, medical, legal, or advisory purposes.

## How to Cite
If you found this educational repository helpful, you can cite it as:

```bibtex
@software{jimmylabs_2026,
  author = {JimmyLabs Contributors},
  title = {JimmyLabs: A From-Scratch Educational GPT},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DevxD98/JimmyLabs}}
}
```
