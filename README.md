<!--
   this readme is half art, half map. the ascii isn't decoration —
   every glyph points at an idea living somewhere in the repo.
-->

<div align="center">

```
        ██╗██╗███╗   ███╗███╗   ███╗██╗   ██╗
        ██║██║████╗ ████║████╗ ████║╚██╗ ██╔╝
        ██║██║██╔████╔██║██╔████╔██║ ╚████╔╝
   ██   ██║██║██║╚██╔╝██║██║╚██╔╝██║  ╚██╔╝
   ╚█████╔╝██║██║ ╚═╝ ██║██║ ╚═╝ ██║   ██║
    ╚════╝ ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝   ╚═╝
        ██╗      █████╗ ██████╗ ███████╗
        ██║     ██╔══██╗██╔══██╗██╔════╝
        ██║     ███████║██████╔╝███████╗
        ██║     ██╔══██║██╔══██╗╚════██║
        ███████╗██║  ██║██████╔╝███████║
        ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝
```

### 🧠 a language model, grown by hand, in 8 gigabytes of silence 🍃

**A GPT-style transformer built from *absolute scratch* — and understood down to the last gradient — on a MacBook Air M1.**

<br/>

![Params](https://img.shields.io/badge/params-1M--4M-8A2BE2?style=for-the-badge)
![Hardware](https://img.shields.io/badge/hardware-M1_Air_8GB-000000?style=for-the-badge&logo=apple)
![Backend](https://img.shields.io/badge/backend-MPS_%2F_Metal-76B900?style=for-the-badge)
![No CUDA](https://img.shields.io/badge/CUDA-not_invited-red?style=for-the-badge)

![Status](https://img.shields.io/badge/status-documentation--first-informational?style=flat-square)
![Philosophy](https://img.shields.io/badge/philosophy-WHY_before_HOW-blueviolet?style=flat-square)
![Timeline](https://img.shields.io/badge/timeline-3–4_months-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

<br/>

> *“every large model was once a small one*
> *that someone refused to stop understanding.”*

[**📖 Read the textbook**](docs/) · [**🗺️ Learning path**](docs/03_LEARNING_PATH.md) · [**🧬 Architecture**](docs/02_ARCHITECTURE.md) · [**⚡ Apple Silicon**](docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) · [**📜 Principles**](ENGINEERING_PRINCIPLES.md)

</div>

---

<div align="center">

```
                     ·  .  ˚    ✦
              .      the void      ˚   ·
           ˚    (untrained weights, pure noise)
                       │
                       │  gradient rain
                       ▼
     ·  ˖  ✦   ┌─────────────────────────────┐   ✦  ˖  ·
               │   t o k e n s   →   m e a n  │
       ˚       └─────────────────────────────┘        ˚
                       │
                       ▼
            ✶  a sentence that was
                never written before  ✶
```

</div>

## ✨ What is this?

**JimmyLabs** is a from-scratch, GPT-style language model — **1–4 million parameters** — engineered, trained, and profiled entirely on a **MacBook Air M1 with 8 GB of unified memory**. No CUDA. No pretrained weights. No prebuilt transformer blocks.

It is **not** trying to beat GPT‑4, Claude, Gemini, or Llama. It's trying to *understand* them — every embedding, every attention head, every gradient — by rebuilding the machine one comprehensible piece at a time, and writing down **why** each piece exists before **how** it works.

> 🧪 This repository is designed to read like a small **research lab**, not a tutorial.
> The code will be tiny. The **understanding** is meant to be enormous.

## 🎛️ The idea in one diagram

```
      raw text                                          new text
         │                                                 ▲
         ▼                                                 │
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────────┐
   │ tokenizer │ → │ embeddings│ → │transformer│ → │  lm head +   │
   │  (chars)  │   │  + pos.   │   │  blocks   │   │  sampling    │
   └───────────┘   └───────────┘   └───────────┘   └──────────────┘
         └──────────── everything below is built by hand ─────────┘
```

<table>
<tr><td>

**✅ We build by hand**

tokenizer · embeddings · positional encoding · multi‑head attention · feed‑forward · LayerNorm · residuals · transformer blocks · GPT · training loop · sampling · checkpointing

</td><td>

**🔌 PyTorch may only do**

tensors · autograd · optimizers · MPS (Metal) acceleration

*…and nothing we could have written ourselves.*

</td></tr>
</table>

## 🚀 Quick start

**Reading first?**

```
   👋 new to ML?     →  docs/03_LEARNING_PATH.md   (assumes basic Python, almost nothing else)
   🧠 know the ropes? →  docs/08_ATTENTION.md        (the conceptual summit)
   🏗️ want the wiring? →  architecture/               (tensor shapes & data flow)
   🔬 here to tinker?  →  research/                    (papers · experiments · decisions)
```

**Running it?** It works — v0.1 trains and generates on a MacBook Air M1. From a clean clone:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install pytest

pytest -q                                          # 43 tests, ~5–10s — proves it all works

python scripts/prepare_data.py                     # downloads + tokenizes tiny Shakespeare
python scripts/train.py --config configs/train_shakespeare.yaml   # ~3 min, val loss 4.17 → ~1.5
python scripts/generate.py --temperature 0.8 --top_k 40            # generate from the checkpoint
```

Extras once trained:

```bash
python scripts/benchmark.py                 # naive generation throughput
python scripts/benchmark.py --use_cache     # KV-cache generation, within block_size (~+65–86% faster; see the correction note below)
python scripts/visualize_attention.py       # ASCII heatmap of what each attention head learned
```

No GPU/CUDA required — MPS (Apple Silicon) is used automatically if available, CPU otherwise.

## 🗂️ Repository map

```
JimmyLabs/
├── 📖 docs/            the textbook (numbered, read in order)
├── 🧬 architecture/    deep-dive wiring diagrams
├── 🔬 research/        paper notes · experiments · benchmarks · design decisions
├── 🧪 experiments/     run logs (one folder per experiment)
├── 📊 benchmarks/      reproducible performance measurements
├── 🗒️ notes/           loose thinking, kept honest
├── 🖼️ assets/          images referenced by docs
├── 📐 diagrams/        source-of-truth ASCII diagrams
├── ⚙️ configs/         annotated model/training configurations
├── 🛠️ scripts/         utility entry points (documented, not yet code)
├── 📚 datasets/        data provenance, licensing, prep notes
├── 🧠 src/             the model itself (module map mirrors architecture/)
├── ✅ tests/           what every module must prove about itself
├── 💾 checkpoints/     saved weights (git-ignored; format documented)
└── 📝 outputs/         generated text samples
```

Every folder has a `README.md` explaining **why it exists**, what belongs inside, and — just as importantly — what does *not*.

## 🧭 Design commitments

```
   ①  understanding  >  speed
   ②  correctness    >  optimization
   ③  measure every optimization — before & after, or it didn't happen
   ④  small, elegant, maintainable code
   ⑤  every experiment reproducible from a config + a seed
```

These aren't slogans — they're enforced by the repo's structure. Every optimization lives next to its benchmark; every architecture change next to its [ADR](research/design_decisions/); every claim next to its evidence. The full constitution: **[ENGINEERING_PRINCIPLES.md](ENGINEERING_PRINCIPLES.md)**.

## 🍎 The machine we build on

```
   ┌──────────────────────────────────────────────────────────┐
   │  MacBook Air M1                                           │
   │  ┌────────────┐   8 GB UNIFIED MEMORY   ┌────────────┐    │
   │  │  CPU cores │ ◀────── one pool ──────▶ │  GPU (MPS) │    │
   │  └────────────┘   no copy across bus     └────────────┘    │
   │                                                            │
   │  the constraint is the point. small memory forces small,   │
   │  honest models — and every optimization must earn its RAM. │
   └──────────────────────────────────────────────────────────┘
```

Apple Silicon is a **first-class target**, not an afterthought → **[docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md](docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md)** · **[architecture/apple_silicon_strategy.md](architecture/apple_silicon_strategy.md)**

## 🤝 Contributing

We welcome contributions, provided they align strictly with the project's educational focus and constitution. Before writing any code, please read our [Engineering Principles](ENGINEERING_PRINCIPLES.md) and review the [Contributing Guide](CONTRIBUTING.md) for our ADR, experiment, and documentation workflows.

## 📡 Status

```
   phase 0  foundations ......  ▓▓▓▓▓▓▓▓▓▓  done — autograd toy · XOR · MNIST
   phase 1  nlp foundations ..  ▓▓▓▓▓▓▓▓▓▓  done — char tokenizer · data loader
   phase 2  tinygpt .........   ▓▓▓▓▓▓▓▓▓▓  done — 0.82M GPT, param-exact
   phase 3  training ........   ▓▓▓▓▓▓▓▓▓▓  done — trains! val loss 4.17 → 1.54
   phase 4  optimization ....   ▓▓▓▓▓▓▓▓▓▓  done — KV cache (see caveat below) · fused AdamW · bf16 rejected
   phase 5  research ........   ▓▓▓▓▓▓▓░░░  ongoing — 6 paper notes · v0.2 trained on TinyStories
```

**v0.1 speaks — and v0.2 speaks more fluently.** v0.1 (0.82M params) generates recognizable
Shakespeare — see [`outputs/trained_shakespeare_sample.txt`](outputs/trained_shakespeare_sample.txt)
next to the pre-training garbage in `untrained_baseline.txt`. v0.2 (2.75M params, trained on
TinyStories) goes further — real names, coherent short sentences, correct dialogue
punctuation — see [`outputs/trained_tinystories_v0_2_sample.txt`](outputs/trained_tinystories_v0_2_sample.txt)
([`benchmarks/004_v0_2.md`](benchmarks/004_v0_2.md): val loss **0.86**, perplexity ≈2.4).
66 tests green, CI-gated.

> **Correction (2026-07-26):** KV-cache generation is correct and fast **within
> `block_size`**; past it, the naive (uncached) path is used instead of a buggy trim —
> see [`docs/21_DEVLOG.md`](docs/21_DEVLOG.md) and
> [`ADR-0004`](research/design_decisions/ADR-0004-no-cache-reuse-past-block-size.md). Cache
> is still **+86%** over naive on the corrected benchmark
> ([`002_kv_cache.md`](benchmarks/002_kv_cache.md) §7).

An active, in-progress **3–4 month educational build**. The documentation is written *ahead of* the code on purpose: **you cannot engineer clearly what you cannot explain clearly.**

<details>
<summary>🎯 <strong>Why documentation-first? (click to expand)</strong></summary>

<br/>

Most hobby ML repos are code-first: someone gets a model training, then maybe writes a README. The understanding — if it ever existed — evaporates. JimmyLabs inverts this. Each component is *explained* before it's *built*, so the code becomes the natural consequence of a clear idea rather than a thing to reverse-engineer later. The docs are the deliverable as much as the model is. See [docs/00_PROJECT_VISION.md](docs/00_PROJECT_VISION.md).

</details>

<details>
<summary>🧩 <strong>Is this really "from scratch"? (click to expand)</strong></summary>

<br/>

Yes — with an honest boundary. We use PyTorch for **tensors, automatic differentiation, optimizers, and MPS acceleration**. We do **not** import prebuilt transformers, attention modules, or tokenizers. Everything in the `text → represent → process → predict` pipeline is hand-written and hand-explained. Autograd is trusted; the architecture is not borrowed.

</details>

---

<div align="center">

```
        ┌─┐┌─┐┌─┐┌─┐  every token you read the model read too.
        └─┘└─┘└─┘└─┘  the difference is it had to guess the next one.
```

<sub><strong>Built by hand on Apple Silicon</strong> · Documentation‑first, forever · [MIT](LICENSE) licensed</sub>

<sub>⭐ if you believe small, understood models still matter.</sub>

</div>
