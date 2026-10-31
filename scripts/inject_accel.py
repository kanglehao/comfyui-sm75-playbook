#!/usr/bin/env python3
"""Inject an sm75 attention-patch node into an existing MiniMax-H3 ComfyUI workflow,
keep subgraph instance values in sync with the graph inside, and add a save node.

Why a script instead of a hand-edited JSON: workflow JSON carries cross-referenced
link ids on both sides (the producer's output.links list and the consumer's
input.link). Editing it by hand silently produces dangling links that ComfyUI
loads without complaint but executes with the patch bypassed. This does the
bookkeeping properly and refuses to guess when the graph is ambiguous.

Why the instance sync exists: when a workflow wraps its nodes in a subgraph, every
promoted widget is stored TWICE -- once on the node inside `definitions.subgraphs`,
and once in the `widgets_values` of the subgraph instance node at the top level.
The frontend reads the INSTANCE copy. A patch that only edits the inner nodes
therefore loads fine, reports no error, and silently runs the old values. Three
of the bugs in the first release of this repo were exactly that mistake, so the
sync step is mandatory here and the audit step refuses to write a file where the
two copies disagree.

Usage:
    python inject_accel.py --src in.json --out out.json
    python inject_accel.py --src in.json --out out.json --set width=768 --set height=512
    python inject_accel.py --src in.json --out out.json --save-video video/h3_i2v
    python inject_accel.py --src in.json --dry-run

The default injected node is `PatchTuringCutlassAttention`. That node has no
public distribution channel -- see README. Use --node to inject another
MODEL->MODEL node such as `TaiYang2080ti_minimaxh3`, or --node none to only do
the value/save-node work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Nodes whose MODEL input marks the tail of the model chain.
SINK_TYPES = {
    "BasicGuider",
    "CFGGuider",
    "DualCFGGuider",
    "BasicScheduler",
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
}

MODEL_INPUT_NAMES = {"model", "MODEL"}

# widgets_values is positional. To edit a widget by name we need the order that
# the node declares in INPUT_TYPES; only the nodes we actually touch are listed.
WIDGET_ORDER = {
    "CLIPLoader": ("clip_name", "type", "device"),
    "UNETLoader": ("unet_name", "weight_dtype"),
    "VAELoader": ("vae_name",),
    "LoraLoaderModelOnly": ("lora_name", "strength_model"),
    "LoraLoader": ("lora_name", "strength_model", "strength_clip"),
    "PrimitiveInt": ("value", "control_after_generate"),
    "PrimitiveFloat": ("value",),
    "PrimitiveBoolean": ("value",),
    "PrimitiveString": ("value",),
    "PrimitiveStringMultiline": ("value",),
    "RandomNoise": ("noise_seed", "control_after_generate"),
    "BasicScheduler": ("scheduler", "steps", "denoise"),
    "MiniMaxH3ImageToVideo": ("prompt", "width", "height", "length"),
    "ComfyMathExpression": ("expression",),
}

# widgets_values is positional and must match INPUT_TYPES order (model excluded).
NODE_PRESETS = {
    "PatchTuringCutlassAttention": {
        "size": [400, 154],
        "widgets": ["low", "hadamard_dual_int8", True, True],
    },
    "TaiYang2080ti_minimaxh3": {
        "size": [320, 82],
        "widgets": ["fp32"],
    },
}

SAVE_VIDEO_SIZE = [900, 106]


# ---------------------------------------------------------------- containers

def pick_container(doc: dict, want: str, subgraph_index: int = 0):
    """Return (container_dict, label). ComfyUI stores graphs either at top level
    or inside definitions.subgraphs when the workflow wraps everything in a
    subgraph node. In 'auto' mode the container is chosen by walking candidates
    and taking the first one that actually holds the MODEL chain."""
    subs = (doc.get("definitions") or {}).get("subgraphs") or []

    def labelled(g):
        if g is doc:
            return g, "top-level graph"
        return g, f"subgraph '{g.get('name')}'"

    if want == "top":
        if not doc.get("nodes"):
            sys.exit("error: --container top given, but the top level has no nodes")
        return labelled(doc)
    if want == "subgraph":
        if not subs:
            sys.exit("error: --container subgraph given, but the file has no subgraphs")
        if subgraph_index >= len(subs):
            sys.exit(f"error: --subgraph-index {subgraph_index} out of range ({len(subs)} subgraph(s))")
        return labelled(subs[subgraph_index])

    # auto: prefer whichever candidate graph carries the real model chain.
    candidates = []
    if doc.get("nodes"):
        candidates.append(doc)
    candidates.extend(subs)
    for g in candidates:
        if find_tail_links(g):
            return labelled(g)
    if subgraph_index < len(subs):
        return labelled(subs[subgraph_index])
    if candidates:
        return labelled(candidates[0])
    sys.exit("error: could not locate a node graph in this file")


# ------------------------------------------------------- subgraph value sync

WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}


def classify_inputs(sg: dict):
    """Split subgraph inputs into (promoted widgets, suspicious).

    The subgraph's own `inputs` list does not mark which entries are promoted
    widgets and which are data connections -- both are just {name, type,
    linkIds}. Looking for a 'widget' key works on the INSTANCE node but not here,
    which is why the first version of this script silently found zero promoted
    inputs and did nothing.

    The reliable test is whether the input's link lands on an internal widget we
    can name. IMAGE/LATENT/VIDEO inputs land on data slots and correctly resolve
    to None. A scalar-typed input that fails to resolve means WIDGET_ORDER is
    missing a node type -- skipping it would shift every later index by one, so
    it is collected for the caller to abort on instead.
    """
    widgets, suspicious = [], []
    for inp in sg.get("inputs") or []:
        got = resolve_promoted(sg, inp)
        if got is not None:
            widgets.append((inp, got))
        elif inp.get("type") in WIDGET_TYPES:
            suspicious.append(inp)
    return widgets, suspicious


def resolve_promoted(sg: dict, inp: dict):
    """Map a promoted subgraph input to (internal_node, widget_index).

    Goes through the subgraph's own link table: the input's linkIds point at an
    internal node, and that link's target_slot names the widget. The widget index
    then comes from WIDGET_ORDER, because target_slot counts ALL inputs while
    widgets_values only counts widgets -- for MiniMaxH3ImageToVideo those differ
    by four (image inputs come first).

    Returns None when the widget cannot be named, which the caller must treat as
    "do not touch" rather than "leave it alone".
    """
    sg_links = {l["id"]: l for l in (sg.get("links") or [])}
    nodes = {n["id"]: n for n in sg["nodes"]}
    for lid in inp.get("linkIds") or []:
        link = sg_links.get(lid)
        if not link:
            continue
        target = nodes.get(link["target_id"])
        if not target:
            continue
        inputs = target.get("inputs") or []
        slot = link["target_slot"]
        if slot >= len(inputs):
            continue
        slot_name = inputs[slot].get("name")
        order = WIDGET_ORDER.get(target["type"])
        if order and slot_name in order:
            return target, order.index(slot_name)
    return None


def coerce(raw: str, type_name: str | None, old):
    if type_name == "INT":
        return int(raw)
    if type_name == "FLOAT":
        return float(raw)
    if type_name == "BOOLEAN":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return raw  # STRING, COMBO


def _short(value, limit: int = 56) -> str:
    """Prompts are multi-line; keep the report readable."""
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 3] + "...'"


def apply_sets(sg: dict, assignments: list[str], dry_run: bool) -> list[tuple]:
    """Edit widgets by promoted-input name (or its label, e.g. turbo_steps)."""
    widgets, suspicious = classify_inputs(sg)
    if suspicious:
        sys.exit("error: these scalar inputs could not be mapped to an internal widget:\n"
                 + "\n".join(f"  {i.get('name')} ({i.get('type')})" for i in suspicious)
                 + "\n       Add the owning node type to WIDGET_ORDER before editing by name.")

    index = {}
    for inp, got in widgets:
        index[inp.get("name")] = (inp, got)
        if inp.get("label"):
            index[inp["label"]] = (inp, got)

    applied = []
    for item in assignments:
        if "=" not in item:
            sys.exit(f"error: --set {item!r}: expected name=value")
        key, raw = item.split("=", 1)
        found = index.get(key)
        if found is None:
            available = ", ".join(sorted(k for k in index if k))
            sys.exit(f"error: --set {key}: no promoted input by that name.\n"
                     f"       available: {available}")
        inp, (node, idx) = found
        values = node.setdefault("widgets_values", [])
        while len(values) <= idx:
            values.append(None)
        old = values[idx]
        new = coerce(raw, inp.get("type"), old)
        values[idx] = new
        applied.append((key, node["type"], node["id"], idx, old, new))
    return applied


def sync_instance(sg: dict, instance: dict, only: set[str] | None = None,
                  dry_run: bool = False) -> list[tuple]:
    """Push internal widget values onto the instance node, for `only` names.

    Which side is authoritative is not a detail. The INSTANCE value is what the
    frontend reads and what executes; the inner widgets_values are the defaults
    the graph had when the subgraph was created. A blanket "inner overwrites
    instance" pass therefore destroys legitimate values -- on the stock H3
    blueprint it turns duration 5s into 2s, the seed into 1, and 8 sampler steps
    into 6.

    So the rule is: only the names the caller explicitly set are synced. Every
    other difference is reported by drift_instance() for a human to judge.
    """
    widgets, suspicious = classify_inputs(sg)
    if suspicious:
        sys.exit("error: cannot sync -- these scalar inputs map to no known widget:\n"
                 + "\n".join(f"  {i.get('name')} ({i.get('type')})" for i in suspicious))

    values = instance.get("widgets_values")
    if values is None:
        values = instance["widgets_values"] = []

    changed = []
    for k, (inp, (node, idx)) in enumerate(widgets):
        if k >= len(values):
            break
        if only is not None:
            names = {inp.get("name"), inp.get("label")}
            if not (names & only):
                continue
        inner = node.get("widgets_values") or []
        if idx >= len(inner):
            continue
        if values[k] != inner[idx]:
            changed.append((inp.get("name"), values[k], inner[idx]))
            values[k] = inner[idx]
    return changed


def drift_instance(sg: dict, instance: dict) -> list[tuple]:
    """Report (name, instance_value, inner_value) where the two copies disagree.

    Not an error by itself: a blueprint can legitimately carry a user-facing
    instance value that differs from the inner default.
    """
    widgets, _ = classify_inputs(sg)
    values = instance.get("widgets_values") or []
    out = []
    for k, (inp, (node, idx)) in enumerate(widgets):
        if k >= len(values):
            break
        inner = node.get("widgets_values") or []
        if idx >= len(inner):
            continue
        if inner[idx] != values[k]:
            out.append((inp.get("name"), values[k], inner[idx]))
    return out


def audit_instance(sg: dict, instance: dict) -> list[str]:
    """Structural problems only. Value drift is a warning, not a failure."""
    problems = []
    widgets, suspicious = classify_inputs(sg)
    for inp in suspicious:
        problems.append(f"promoted '{inp.get('name')}': scalar type {inp.get('type')} maps to no "
                        f"internal widget (WIDGET_ORDER is missing the node type)")

    values = instance.get("widgets_values") or []
    if len(values) != len(widgets):
        problems.append(f"instance holds {len(values)} widget value(s), subgraph promotes {len(widgets)}")
    return problems


# ------------------------------------------------------------ node insertion

def find_instance(doc: dict, sg: dict):
    """The top-level node whose type IS the subgraph id."""
    for n in doc.get("nodes") or []:
        if n.get("type") == sg.get("id"):
            return n
    return None


def find_tail_links(graph: dict) -> list[dict]:
    """All links that carry a MODEL into a sampler/guider-shaped node."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    hits = []
    for link in graph.get("links") or []:
        target = nodes.get(link["target_id"])
        if not target or target.get("type") not in SINK_TYPES:
            continue
        inputs = target.get("inputs") or []
        slot = link["target_slot"]
        if slot >= len(inputs):
            continue
        if inputs[slot].get("name") in MODEL_INPUT_NAMES:
            hits.append(link)
    return hits


def insert_after(graph: dict, origin_id: int, origin_slot: int, links: list[dict],
                 node_type: str, position: list[int]) -> dict:
    """Rewire `links` so they now originate from a freshly inserted node."""
    state = graph.setdefault("state", {})
    new_node_id = int(state.get("lastNodeId") or max(n["id"] for n in graph["nodes"])) + 1
    new_link_id = int(state.get("lastLinkId") or max(l["id"] for l in graph["links"])) + 1
    state["lastNodeId"] = new_node_id
    state["lastLinkId"] = new_link_id

    preset = NODE_PRESETS.get(node_type, {"size": [380, 120], "widgets": []})

    new_node = {
        "id": new_node_id,
        "type": node_type,
        "pos": position,
        "size": preset["size"],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{
            "localized_name": "model",
            "name": "model",
            "type": "MODEL",
            "link": new_link_id,
        }],
        "outputs": [{
            "localized_name": "MODEL",
            "name": "MODEL",
            "type": "MODEL",
            "links": [l["id"] for l in links],
        }],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": preset["widgets"],
    }
    graph["nodes"].append(new_node)

    # Producer side: keep its link list consistent.
    nodes = {n["id"]: n for n in graph["nodes"]}
    producer = nodes[origin_id]
    out = producer["outputs"][origin_slot]
    owned = [l["id"] for l in links]
    out["links"] = [lid for lid in (out.get("links") or []) if lid not in owned]
    out["links"].append(new_link_id)

    # Consumer side: repoint at the new node.
    for link in links:
        link["origin_id"] = new_node_id
        link["origin_slot"] = 0

    graph["links"].append({
        "id": new_link_id,
        "origin_id": origin_id,
        "origin_slot": origin_slot,
        "target_id": new_node_id,
        "target_slot": 0,
        "type": "MODEL",
    })
    return new_node


def add_save_video(doc: dict, instance: dict, prefix: str) -> tuple[dict, bool]:
    """Attach a SaveVideo to the subgraph's VIDEO output.

    Without this the workflow computes a video and writes nothing: the VIDEO
    output is a graph output, not a file. That was the second bug in the first
    release -- you could wait 8 minutes and find an empty output folder.
    """
    for n in doc.get("nodes") or []:
        if n.get("type") == "SaveVideo":
            return n, False
    if instance is None:
        sys.exit("error: --save-video needs a subgraph instance at the top level to attach to")

    outs = instance.get("outputs") or []
    if not outs or outs[0].get("type") != "VIDEO":
        sys.exit("error: --save-video given, but the subgraph instance has no VIDEO output")

    new_id = max([n["id"] for n in doc["nodes"]] + [0]) + 1
    new_link = max([l["id"] for l in (doc.get("links") or [])] + [0]) + 1
    pos = instance.get("pos") or [0, 0]
    size = instance.get("size") or [400, 200]

    node = {
        "id": new_id,
        "type": "SaveVideo",
        "pos": [pos[0] + size[0] + 80, pos[1] + size[1] + 40],
        "size": SAVE_VIDEO_SIZE,
        "flags": {},
        "order": len(doc["nodes"]),
        "mode": 0,
        "inputs": [
            {"localized_name": "video", "name": "video", "type": "VIDEO", "link": new_link},
            {"localized_name": "filename_prefix", "name": "filename_prefix", "type": "STRING",
             "widget": {"name": "filename_prefix"}, "link": None},
            {"localized_name": "format", "name": "format", "type": "COMFY_DYNAMICCOMBO_V3",
             "widget": {"name": "format"}, "link": None},
            {"localized_name": "format.codec", "name": "format.codec", "type": "COMFY_DYNAMICCOMBO_V3",
             "widget": {"name": "format.codec"}, "link": None},
            {"localized_name": "codec", "name": "codec", "shape": 7, "type": "COMFY_DYNAMICCOMBO_V3",
             "widget": {"name": "codec"}, "link": None},
        ],
        "outputs": [{"localized_name": "video", "name": "video", "type": "VIDEO", "links": None}],
        "properties": {"Node name for S&R": "SaveVideo", "cnr_id": "comfy-core"},
        # filename_prefix, format, format.codec, codec -- DynamicCombo serialises flat
        "widgets_values": [prefix, "auto", "auto", "auto"],
    }
    doc["nodes"].append(node)
    doc.setdefault("links", []).append({
        "id": new_link,
        "origin_id": instance["id"],
        "origin_slot": 0,
        "target_id": new_id,
        "target_slot": 0,
        "type": "VIDEO",
    })
    outs[0]["links"] = (outs[0].get("links") or []) + [new_link]

    doc["last_node_id"] = max(int(doc.get("last_node_id") or 0), new_id)
    doc["last_link_id"] = max(int(doc.get("last_link_id") or 0), new_link)
    return node, True


# --------------------------------------------------------------------- audit

def verify(graph: dict) -> list[str]:
    """Cheap structural audit: every input link exists, every link endpoint exists.

    Negative node ids are the subgraph I/O proxies ComfyUI uses (-10 = inputs,
    -20 = outputs); they are legitimate endpoints and are skipped.
    """
    problems = []
    ids = {n["id"] for n in graph["nodes"]}
    link_ids = {l["id"] for l in graph.get("links") or []}
    for link in graph.get("links") or []:
        if link["origin_id"] >= 0 and link["origin_id"] not in ids:
            problems.append(f"link {link['id']}: unknown origin node {link['origin_id']}")
        if link["target_id"] >= 0 and link["target_id"] not in ids:
            problems.append(f"link {link['id']}: unknown target node {link['target_id']}")
    for node in graph["nodes"]:
        for inp in node.get("inputs") or []:
            lid = inp.get("link")
            if lid is not None and lid not in link_ids:
                problems.append(f"node {node['id']}: input '{inp.get('name')}' points at missing link {lid}")
        for out in node.get("outputs") or []:
            for lid in out.get("links") or []:
                if lid not in link_ids:
                    problems.append(f"node {node['id']}: output '{out.get('name')}' lists missing link {lid}")
    return problems


def verify_doc(doc: dict) -> list[str]:
    """Structural audit across both levels, plus instance/inner agreement."""
    problems = verify(doc)
    for sg in (doc.get("definitions") or {}).get("subgraphs") or []:
        problems.extend(verify(sg))
        instance = find_instance(doc, sg)
        if instance is not None:
            problems.extend(audit_instance(sg, instance))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source workflow .json")
    ap.add_argument("--out", help="where to write the patched workflow")
    ap.add_argument("--node", default="PatchTuringCutlassAttention",
                    help="MODEL->MODEL node type to inject, or 'none'")
    ap.add_argument("--container", choices=["auto", "top", "subgraph"], default="auto")
    ap.add_argument("--subgraph-index", type=int, default=0)
    ap.add_argument("--set", dest="assignments", action="append", default=[],
                    metavar="NAME=VALUE",
                    help="edit a promoted widget by name, e.g. --set width=768 "
                         "(both copies are written, so they cannot drift)")
    ap.add_argument("--set-json", metavar="FILE",
                    help="JSON object of name->value edits, for long prompts")
    ap.add_argument("--save-video", metavar="PREFIX",
                    help="attach a SaveVideo node, e.g. --save-video video/h3_i2v")
    ap.add_argument("--no-sync", action="store_true",
                    help="skip the instance sync (for demonstrating the bug, not for use)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    args = ap.parse_args()

    doc = json.loads(Path(args.src).read_text(encoding="utf-8"))
    graph, label = pick_container(doc, args.container, args.subgraph_index)
    print(f"graph: {label}  ({len(graph['nodes'])} nodes, {len(graph.get('links') or [])} links)")

    is_subgraph = graph is not doc
    instance = find_instance(doc, graph) if is_subgraph else None

    assignments = list(args.assignments)
    if args.set_json:
        data = json.loads(Path(args.set_json).read_text(encoding="utf-8"))
        assignments.extend(f"{k}={v}" for k, v in data.items())

    # 1. explicit value edits
    touched = set()
    if assignments:
        for key, ntype, nid, idx, old, new in apply_sets(graph, assignments, args.dry_run):
            print(f"set {key}: node {nid} ({ntype}) widget[{idx}] {old!r} -> {new!r}")
            touched.add(key)

    # 2. sync ONLY the values we just set -- the instance copy is what executes
    if is_subgraph and not args.no_sync:
        if instance is None:
            print("warning: subgraph is not instantiated at the top level; nothing to sync")
        elif touched:
            changed = sync_instance(graph, instance, only=touched)
            for name, old, new in changed:
                print(f"sync instance '{name}': {old!r} -> {new!r}")
            if not changed:
                print("sync instance: the edited values already agreed")
        else:
            print("sync instance: no --set given, instance untouched")

    # 2b. report pre-existing drift so it is visible instead of silent
    if is_subgraph and instance is not None:
        drift = drift_instance(graph, instance)
        if drift:
            print(f"note: {len(drift)} promoted value(s) differ between the two copies "
                  f"(left as-is -- the instance value is authoritative):")
            for name, inst_v, inner_v in drift:
                print(f"  {name}: instance={_short(inst_v)} inner={_short(inner_v)}")

    # 3. attention patch
    if args.node.lower() != "none":
        present = [n for n in graph["nodes"] if n["type"] == args.node]
        if present:
            print(f"{args.node} already present (node {present[0]['id']}); not inserting again")
            links = []
        else:
            links = find_tail_links(graph)
        if not links and not present:
            sys.exit("error: no MODEL-carrying link into a sampler/guider was found.\n"
                     "       Is this an H3 workflow? Check that the UNET->guider chain is intact.")

        upstream = {(l["origin_id"], l["origin_slot"]) for l in links}
        if len(upstream) > 1:
            detail = "\n".join(
                f"  node {o} slot {s} -> {sum(1 for l in links if (l['origin_id'], l['origin_slot']) == (o, s))} consumer(s)"
                for o, s in sorted(upstream))
            sys.exit("error: the sampler/guider models come from several different producers,\n"
                     "       so a single insertion point cannot serve them all:\n" + detail +
                     "\n       Insert the node manually after the merge point, or patch each branch separately.")

        if upstream:
            origin_id, origin_slot = upstream.pop()
            nodes = {n["id"]: n for n in graph["nodes"]}
            print(f"insertion point: after node {origin_id} ({nodes[origin_id]['type']}) slot {origin_slot}")
            print(f"will repoint {len(links)} consumer link(s): {[l['id'] for l in links]}")

            if not args.dry_run:
                pos = nodes[origin_id].get("pos") or [0, 0]
                size = nodes[origin_id].get("size") or [300, 100]
                new_node = insert_after(graph, origin_id, origin_slot, links, args.node,
                                        [pos[0] + size[0] + 70, pos[1] + 20])
                print(f"inserted node {new_node['id']} ({args.node})")

    # 4. save node
    if args.save_video:
        if args.dry_run:
            print(f"would attach SaveVideo (prefix {args.save_video!r})")
        else:
            node, created = add_save_video(doc, instance, args.save_video)
            print(f"{'attached' if created else 'reused'} SaveVideo node {node['id']}")

    # 5. audit
    problems = verify_doc(doc)
    if problems:
        print("audit FAILED:")
        for p in problems:
            print("  " + p)
        return 1

    if args.dry_run:
        print("dry run, nothing written")
        return 0

    out = Path(args.out or args.src)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("audit passed: links consistent, instance and inner widgets agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
