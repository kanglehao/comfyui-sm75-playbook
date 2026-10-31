# Changelog

All notable changes to this repository. Dates are the measurement date, not the
publication date — the numbers in the docs were taken before they were written down.

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
