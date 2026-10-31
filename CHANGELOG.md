# Changelog

All notable changes to this repository. Dates are the measurement date, not the
publication date — the numbers in the docs were taken before they were written down.

## [1.0.0] — 2026-10-31

First consolidated release. Everything here came out of one machine
(dual RTX 2080 Ti 22G, Windows, ComfyUI 0.37.0) over roughly six weeks.

### Added

- `README.md` — the short version: what works, what doesn't, what it costs
- `docs/00-hardware.md` — sm_75 boundary, TU102 unlock and why it barely helps video
- `docs/01-environment.md` — pinned version matrix + which packages are "installed but unused"
- `docs/02-launch-flags.md` — every flag with its purpose and side effect
- `docs/03-acceleration-stack.md` — the five layers, node wiring, parameter values
- `docs/03b-sageattention.md` — why stock sageattention 2 fails on sm75, what the patched build does
- `docs/04-benchmarks.md` — A/B numbers, stage breakdown, and how to measure without fooling yourself
- `docs/05-pitfalls.md` — six traps, including the convrot cold-start failure
- `docs/06-models.md` — model list with formats, sizes and sources
- `docs/07-workflow-wiring.md` — insertion rules and the two common miswirings
- `workflows/h3-i2v-2080ti-cutlass.json` — official H3 I2V blueprint + CUTLASS node, retuned for 22 GB
- `scripts/check_env.py` — environment self-check with a paste-ready report
- `scripts/inject_accel.py` — inject the acceleration node into any H3 workflow, with structural audit

### Findings worth remembering

- The `int8_convrot` dequant crash is a **cold-start first-launch failure**, not a
  memory/alignment/stream problem. Warm-up fixes it; retry does not.
- Veda's 3.16× is real but **only on single-card dense-attention baselines**.
  Against the dual-card engine it is noise or worse.
- End-to-end time is **63% VAE decode**. Further attention work has a low ceiling.
- `--vram-headroom` reliably causes `illegal memory access` on this setup. It raises
  the aimdo floor instead of lowering the peak.
- Global `--use-sage-attention` is **short-circuited** by any node that installs
  `optimized_attention_override`. The two cannot be combined.

### Known gaps

- No measured numbers for PDD distillation on sm75 yet (node exists, not benchmarked here)
- No second-pass (SelfLift) numbers — it OOMs on dual card
- Sol-VAE untested beyond "ComfyUI 0.37 rejects it"
- Linux numbers are from a different machine class; the docs mark which is which
