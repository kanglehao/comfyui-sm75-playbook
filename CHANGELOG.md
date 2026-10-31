# Changelog

All notable changes to this repository. Dates are the measurement date, not the
publication date — the numbers in the docs were taken before they were written down.

## [2.1.0] — 2026-10-31

Everything here comes from one reader who actually ran the playbook on 0.43.2,
single 2080 Ti, and reported back. He was right about most of it, including one
mistake I had documented and then committed myself.

### Fixed

- **The shipped workflow contradicted the rule it shipped with.** `docs/11` warns
  that a subgraph stores every promoted widget twice and the frontend reads the
  INSTANCE copy — then `h3-i2v-2080ti-cutlass.json` shipped with the inner graph
  set to 768×512 + `int4_convrot` while the instance still said 1344×768 +
  `nvfp4_awq`, empty prompt. Loads fine, runs the wrong thing.
  `inject_accel.py` now writes both copies and refuses to emit a file where they
  disagree.
- **No save node.** The subgraph's VIDEO output is a graph output, not a file.
  The workflow computed a video and wrote nothing. Added `SaveVideo` at the top
  level; `--save-video` attaches it automatically.
- **LoRA source missing from the download table**, and the `_comfyui_` suffix was
  never explained as mandatory. Same filename without it loads PEFT-keyed weights
  that match nothing: no error, no warning, no acceleration. Sizes tell them
  apart (1,956,193,000 vs 1,383,677,808 bytes).
- `--cuda-device all` appeared in the "minimal set" at the top of `docs/02` and
  then vanished from the full command at the bottom. Top section now splits
  single-card vs dual-card properly.

### Corrected

- **Pitfall #1: the patch does not cover the hot path, and hasn't since 0.41.3.**
  The reader saw one call per run on 0.43.2; the same is true on this repo's own
  baseline. `comfy/ops.py:1071` runs `ck.int8_linear(..., convrot=True)` and never
  dequantizes; `dequantize_int8_convrot_weight_dtype` is only reached from
  `TensorWiseINT8Layout.dequantize()`. The 409/1101 counts came from versions
  where dequantization *was* the per-step path. Section rewritten as a
  version-gated check, and `scripts/check_patch_reach.py` now answers "is this
  hook doing anything on my build" with an exit code.
- The rowwise claim "only fastvideo quantizes token_refiner" is now phrased so it
  no longer implies other models are safe. Frequency differs; the hardening
  recommendation does not.
- `docs/07` said "only three changes" to the blueprint. There were six, two of
  which were absent rather than changed.
- `--disable-dynamic-vram` was credited with more than the evidence supports. It
  is now marked optional, with the missing ablation stated plainly.
- `check_env.py` prints presence only and says so — it never proved a workflow
  could load anything.

### Added

- `docs/13-third-party-verification.md` — his measurements (491 / 163 / 427 s at
  768×512×56f, 19.4 s/it steady), the ten issues he raised with my per-item
  verdict, and an explicit list of what I did **not** re-verify.
- `scripts/check_patch_reach.py` — static call-chain check, exit 0/1/2.
- `workflows/set-768x512-int4.json` — the `--set-json` input used to generate the
  shipped workflow, so the derivation is reproducible rather than described.

### Not fixed, deliberately

- `PatchTuringCutlassAttention` still has no public distribution channel. The
  README now says so above the fold instead of burying the dependency. The
  binary is **not** mirrored here: no LICENSE, compiled `.pyd` only, no upstream
  repository, no releases. Its mechanism is documented so the node can be
  reimplemented (attention override + `set_model_attn1_replace`), and the MLP
  chunking half does not need that binary at all.

## [2.0.0] — 2026-10-31

Scope widened from "H3 acceleration" to "everything that had to be fixed to make
this machine usable". Acceleration was maybe a third of the work; the rest was
frontend regressions, workflow repair, and dependency archaeology.

Renamed to `comfyui-sm75-playbook` to match.

### Added

- `docs/10-frontend-compat.md` — four ways the frontend upgrade broke things:
  circuit-board link geometry, double-applied dpr, Impact-Pack load abort,
  `structuredClone` failure. Each with symptom, root cause, and a fix that
  usually does **not** require a backend restart.
- `docs/11-workflow-repair.md` — the backslash rule, subgraph dual widget storage,
  and the five-step verification that stops at nothing short of `graphToPrompt()`.
- `docs/12-deps-and-download.md` — the two environment-level import failures
  (transformers 5.x → peft, albucore → insightface), `--no-deps` discipline,
  HF mirror measurements, and why killing a download can cost you 20 minutes.
- `patches/` — the two actual code contributions: `h3_convrot_warmup` and
  `h3_int8_guard`. Both are update-proof hooks that live in `custom_nodes/`
  rather than `site-packages/`.

### Corrected

- Version matrix now reflects the **real** production instance
  (ComfyUI 0.41.3 / comfy-kitchen 0.2.37 / comfy-aimdo 0.5.6).
  An earlier draft was written against a stale 0.37.0 instance — the numbers
  happened to be close, but the reasoning was based on the wrong tree.
- Pitfall #1 no longer claims the convrot failure is a "cold-start first-launch"
  problem. That was the 2026-09-19 conclusion; 2026-09-21 disproved it
  (warm-up succeeded, crash moved to call #409). Current write-up states only
  what the evidence supports: static params legal, same input re-sent succeeds,
  failure point not fixed.
- SageAttention 2 section now leads with the real blocker — Triton emits no
  tensor-core instructions at `cc < 8` (`mma.sync` count = 0) — instead of
  framing it as "needs a patched wheel".
- Veda marked as requiring a **self-written sm75 backend**; upstream explicitly
  does not support SM75. Anyone cloning upstream Veda will get a hardware
  rejection, so this had to be stated plainly.

### Findings worth remembering

- 63% of end-to-end time is VAE decode. Further attention work is a dead end
  on this machine until Sol-VAE becomes usable.
- Veda's 3.16× is real only against a single-card dense-attention baseline.
  Against the dual-card engine it is noise or worse.
- The same `invalid argument` class of failure hits **two different native ops**,
  and the second one only fires on models that quantize `token_refiner`.
  Hence two patches, not one.
- Global `--use-sage-attention` is short-circuited by anything that installs
  `optimized_attention_override`. Combining them is a no-op, not a conflict.

### Known gaps

- No measured PDD distillation numbers on sm75 (node installed, not benchmarked)
- No second-pass (SelfLift) numbers — OOM on dual card
- Veda sm75 backend A/B across end-to-end runs not measured; only per-kernel
- Linux numbers come from a different machine class and are labeled as such
