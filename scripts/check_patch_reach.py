#!/usr/bin/env python3
"""Is the convrot warm-up/retry patch still covering a path that actually runs?

The patch in patches/h3_convrot_warmup hooks
`comfy_kitchen...dequantize_int8_convrot_weight_dtype`. Whether that hook ever
fires during sampling depends on the ComfyUI/kitchen version, and it stopped
firing at some point without anyone announcing it:

  - older kitchen: an INT8 convrot linear dequantized its weight first, then ran
    a normal GEMM. Dequantization was on the hot path, once per linear per step,
    which is why the failure counters reached 409 and 1101.
  - kitchen >= the version that added `int8_linear`: the GEMM consumes the
    quantized weight directly and never materializes the dequantized copy.
    Dequantization survives only on explicit paths (patching a LoRA into an
    INT8 weight, casting, saving), so the hook fires a handful of times per run
    instead of thousands.

A patched-in warm-up that fires a handful of times is not protecting anything.
It also fails safely: logs are clean, inference works, and the thing looks
fixed. That is the worst outcome, so check instead of assuming.

Run with the SAME interpreter that runs ComfyUI:

    python scripts/check_patch_reach.py --comfyui /path/to/ComfyUI

Exit codes: 0 = hot path still dequantizes (patch is doing work),
            1 = hot path bypasses dequantization (patch is decorative),
            2 = could not determine.
"""

from __future__ import annotations

import argparse
import re
import sys
import sysconfig
from pathlib import Path


def find_site_packages() -> Path | None:
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p and Path(p).is_dir():
            return Path(p)
    return None


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def kitchen_version(sp: Path) -> str:
    for d in sorted(sp.glob("comfy_kitchen-*.dist-info")):
        return d.name.split("comfy_kitchen-", 1)[1].split(".dist-info")[0]
    return "unknown"


def count_static_callers(sp: Path, func: str) -> list[str]:
    """Files that call func, excluding its own definitions and internal forwards.

    Kitchen routes the same op through several backends (cuda / eager / hip), so
    a call site inside comfy_kitchen is usually a forwarder, not a consumer. They
    are still listed -- the point is to show that NONE of them is in comfy/.
    """
    hits = []
    root = sp / "comfy_kitchen"
    if not root.is_dir():
        return hits
    pat = re.compile(rf"(?<!def ){re.escape(func)}\s*\(")
    for py in root.rglob("*.py"):
        for line in read(py).splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("def ") and func in stripped:
                continue
            if pat.search(line):
                rel = py.relative_to(sp)
                hits.append(f"{rel}: {stripped[:88]}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfyui", required=True, help="ComfyUI root (the directory holding main.py)")
    args = ap.parse_args()

    sp = find_site_packages()
    if sp is None:
        print("error: cannot locate site-packages for this interpreter")
        return 2

    comfy = Path(args.comfyui)
    ops = comfy / "comfy" / "ops.py"
    if not ops.is_file():
        print(f"error: {ops} not found -- is --comfyui pointing at the ComfyUI root?")
        return 2

    print(f"interpreter   {sys.executable}")
    print(f"site-packages {sp}")
    print(f"comfy_kitchen {kitchen_version(sp)}")
    print()

    # 1. does this kitchen expose the fused kernel?
    cuda_backend = sp / "comfy_kitchen" / "backends" / "cuda" / "__init__.py"
    backend_text = read(cuda_backend)
    has_int8_linear = bool(re.search(r"^def int8_linear\(", backend_text, re.M))
    print(f"fused int8_linear in kitchen : {'yes' if has_int8_linear else 'no'}")

    # 2. does ComfyUI call it?
    ops_text = read(ops)
    ops_calls_int8 = "int8_linear(" in ops_text
    print(f"comfy/ops.py calls int8_linear : {'yes' if ops_calls_int8 else 'no'}")

    # 3. where else does dequantization get called from?
    deq_dtype_callers = count_static_callers(sp, "dequantize_int8_convrot_weight_dtype")
    print(f"static callers of dequantize_int8_convrot_weight_dtype : {len(deq_dtype_callers)}")
    for line in deq_dtype_callers[:6]:
        print(f"    {line}")

    # 4. does ComfyUI itself dequantize before its GEMMs?
    comfy_deq = []
    for py in (comfy / "comfy").rglob("*.py"):
        if "dequantize_int8_convrot_weight" in read(py):
            comfy_deq.append(str(py.relative_to(comfy)))
    print(f"comfy/** references convrot dequantize : {len(comfy_deq)}"
          + (f" ({', '.join(comfy_deq[:3])})" if comfy_deq else ""))

    print()
    if has_int8_linear and ops_calls_int8 and not comfy_deq:
        print("VERDICT: the sampling path consumes quantized weights directly and never")
        print("         dequantizes them. The warm-up/retry hook cannot fire on the")
        print("         per-step path -- expect a handful of calls per run (LoRA patching),")
        print("         not hundreds per step.")
        print()
        print("         The patch is harmless but it is NOT the thing keeping you stable.")
        print("         If you still see 'convrot dequantization failed', the failing call")
        print("         is somewhere else; capture a stack trace at the failure instead of")
        print("         widening the retry.")
        return 1

    if not has_int8_linear:
        print("VERDICT: this kitchen has no fused int8_linear, so convrot linears most")
        print("         likely dequantize before their GEMM. The hook sits on the hot path")
        print("         and the patch is doing real work.")
        return 0

    print("VERDICT: inconclusive. int8_linear exists but comfy/ops.py does not call it,")
    print("         or something else still dequantizes. Read the call sites above.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
