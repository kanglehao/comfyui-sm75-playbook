#!/usr/bin/env python3
"""Report whether this machine matches the sm75 recipe in this repository.

Run it with the SAME Python that launches ComfyUI (portable builds ship their
own interpreter, so a system python will report the wrong torch):

    E:\\...\\python_embeded\\python.exe scripts\\check_env.py
    python scripts/check_env.py --comfyui /path/to/ComfyUI

Nothing is modified. The output is meant to be pasted into an issue.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

# (package, attribute that must exist, expected version or None for "informational")
EXPECTED = {
    "torch": None,
    "torchvision": None,
    "comfy_kitchen": ">=0.2.33",
    "comfy_aimdo": "informational",
    "sageattention": "informational",
    "triton": "informational",
    "flash_attn": "informational",
}

REQUIRED_NODES = {
    "comfyui-minimax-h3-turing": "Patch Turing CUTLASS Attention (sm75)",
}

OPTIONAL_NODES = {
    "ComfyUI-H3-MultiGPU": "dual-GPU engine (H3VM)",
    "ComfyUI-MiniMax-H3-PDD-Acc": "8/4-step PDD distillation",
    "h3_convrot_warmup": "convrot cold-start warm-up",
}

REQUIRED_MODELS = {
    "diffusion_models/MiniMax-H3/minimax_h3_fl2va_pruned_int8_convrot.safetensors": "UNET (int8, 56 heads)",
    "text_encoders/MiniMax-H3/qwen3vl_32b_minimax_h3_int4_convrot.safetensors": "text encoder (int4)",
    "vae/MiniMax-H3/minimax_h3_video_vae_fp16.safetensors": "video VAE",
    "vae/MiniMax-H3/minimax_h3_audio_vae_fp32.safetensors": "audio VAE",
}

# Files that must NOT be used on sm75. Presence is reported as a warning, since
# people often have several variants side by side.
FORBIDDEN_MODEL_HINTS = ("nvfp4",)


def vram_gib() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return None


def free_vram_gib() -> float | None:
    """Currently free VRAM. On a 22G modded card this reads ~21.6 GiB when idle,
    which is the number the engine has to work with, not the 22.0 total."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(0)
        return free / (1024 ** 3)
    except Exception:
        return None


def pkg_version(name: str) -> str | None:
    """Prefer installed-distribution metadata: many of these packages ship no
    __version__ attribute, and the dist name may use a hyphen while the import
    name uses an underscore."""
    if importlib.util.find_spec(name) is None:
        return None
    try:
        from importlib.metadata import PackageNotFoundError, version
        for candidate in (name, name.replace("_", "-")):
            try:
                return version(candidate)
            except PackageNotFoundError:
                continue
    except Exception:
        pass
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception as e:  # some of these fail to import on the wrong arch
        return f"import failed: {type(e).__name__}"


def find_comfyui(start: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if (p / "main.py").is_file() else None
    for parent in [start.parent, *start.parents]:
        if (parent / "main.py").is_file() and (parent / "comfy").is_dir():
            return parent
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", help="path to the ComfyUI directory")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    here = Path(__file__).resolve()
    comfy = find_comfyui(here, args.comfyui)
    report: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "comfyui_dir": str(comfy) if comfy else None,
        "warnings": [],
        "problems": [],
    }

    # ---- hardware -------------------------------------------------------
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            report["gpu"] = torch.cuda.get_device_name(0)
            report["capability"] = f"sm_{cap[0]}{cap[1]}"
            report["gpu_count"] = torch.cuda.device_count()
            report["vram_gib"] = round(vram_gib() or 0, 2)
            report["free_vram_gib"] = round(free_vram_gib() or 0, 2)
            report["is_turing"] = cap == (7, 5)
            if cap != (7, 5):
                report["warnings"].append(
                    f"capability is {cap}, not (7,5). This recipe is sm75-specific; "
                    "on sm80+ use sageattention/FlashAttention instead of the CUTLASS node."
                )
            if (report["vram_gib"] or 0) < 11:
                report["problems"].append(
                    f"only {report['vram_gib']} GiB VRAM; H3 int8 + 32B text encoder needs ~11 GiB minimum"
                )
    except ImportError:
        report["problems"].append("torch not importable — run this with ComfyUI's embedded python")

    # ---- packages -------------------------------------------------------
    pkgs = {}
    for name in EXPECTED:
        v = pkg_version(name)
        pkgs[name] = v
    report["packages"] = pkgs

    ck = pkgs.get("comfy_kitchen")
    if ck is None:
        report["problems"].append("comfy-kitchen missing — int8_convrot dequantization will not work")
    elif ck.startswith("0.2.") and ck < "0.2.33":
        report["problems"].append(f"comfy-kitchen {ck} is older than 0.2.33; upgrade")

    if pkgs.get("flash_attn"):
        report["warnings"].append(
            f"flash_attn {pkgs['flash_attn']} present but sm75 has no compatible kernels; it will not be used"
        )

    # ---- custom nodes ---------------------------------------------------
    if comfy:
        nodes_dir = comfy / "custom_nodes"
        report["custom_nodes"] = {}
        for name, purpose in REQUIRED_NODES.items():
            ok = (nodes_dir / name).is_dir()
            report["custom_nodes"][name] = ok
            if not ok:
                report["problems"].append(f"missing custom node: {name} ({purpose})")
        for name, purpose in OPTIONAL_NODES.items():
            ok = (nodes_dir / name).is_dir()
            report["custom_nodes"][name] = ok
            if not ok:
                report["warnings"].append(f"optional custom node not found: {name} ({purpose})")

        # ---- models -----------------------------------------------------
        models_dir = comfy / "models"
        report["models"] = {}
        for rel, purpose in REQUIRED_MODELS.items():
            p = models_dir / rel
            exists = p.is_file()
            size = round(p.stat().st_size / (1024 ** 3), 2) if exists else None
            report["models"][rel] = {"present": exists, "size_gib": size}
            if not exists:
                report["problems"].append(f"missing model: {rel} ({purpose})")

        # nvfp4 files are harmless to have on disk, but must not be selected.
        nvfp4 = [str(p.relative_to(models_dir)) for p in models_dir.rglob("*nvfp4*") if p.is_file()]
        if nvfp4:
            report["warnings"].append(
                "nvfp4 weights found — do NOT select them on sm75 (no FP4 hardware): "
                + ", ".join(nvfp4[:5])
            )
    else:
        report["problems"].append(
            "could not locate ComfyUI (no main.py found upward from this script); pass --comfyui"
        )

    # ---- verdict --------------------------------------------------------
    if report["problems"]:
        verdict = "NOT READY"
    elif report["warnings"]:
        verdict = "READY (with warnings)"
    else:
        verdict = "READY"
    report["verdict"] = verdict

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if not report["problems"] else 1

    line = "=" * 62
    print(line)
    print(" ComfyUI sm75 acceleration — environment check")
    print(line)
    print(f" verdict        {verdict}")
    print(f" python         {report['python']}")
    print(f" platform       {report['platform']}")
    if report.get("gpu"):
        print(f" gpu            {report['gpu']} x{report.get('gpu_count', 1)}")
        print(f" capability     {report['capability']}")
        print(f" vram           {report['vram_gib']} GiB total, "
              f"{report.get('free_vram_gib', report['vram_gib'])} GiB free now")
        print(f" torch          {report.get('torch')}  (cuda {report.get('cuda_runtime')})")
    print()
    print(" packages")
    for name, v in pkgs.items():
        print(f"   {name:<18} {v if v else '-'}")
    if "custom_nodes" in report:
        print()
        print(" custom nodes")
        for name, ok in report["custom_nodes"].items():
            print(f"   {'OK ' if ok else '-- '} {name}")
    if "models" in report:
        print()
        print(" models")
        for rel, info in report["models"].items():
            mark = "OK " if info["present"] else "-- "
            size = f"{info['size_gib']} GiB" if info["size_gib"] else ""
            print(f"   {mark}{rel}  {size}")
    if report["problems"]:
        print()
        print(" problems")
        for p in report["problems"]:
            print(f"   ! {p}")
    if report["warnings"]:
        print()
        print(" warnings")
        for w in report["warnings"]:
            print(f"   ~ {w}")
    print(line)
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
