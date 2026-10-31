#!/usr/bin/env python3
"""Inject an sm75 attention-patch node into an existing MiniMax-H3 ComfyUI workflow.

Why a script instead of a hand-edited JSON: workflow JSON carries cross-referenced
link ids on both sides (the producer's output.links list and the consumer's
input.link). Editing it by hand silently produces dangling links that ComfyUI
loads without complaint but executes with the patch bypassed. This does the
bookkeeping properly and refuses to guess when the graph is ambiguous.

Insertion point: the MODEL wire(s) that feed the sampler/guider, i.e. the very
tail of the model chain. That is where every H3 workflow ends up regardless of
whether it uses a LoRA, a switch, or neither.

Usage:
    python inject_accel.py --src in.json --out out.json
    python inject_accel.py --src in.json --out out.json --container subgraph
    python inject_accel.py --src in.json --dry-run

The default injected node is `PatchTuringCutlassAttention`
(custom_nodes/comfyui-minimax-h3-turing). Use --node to inject another
MODEL->MODEL node such as `TaiYang2080ti_minimaxh3`.
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


def find_tail_links(graph: dict) -> list[dict]:
    """All links that carry a MODEL into a sampler/guider-shaped node."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    hits = []
    for link in graph["links"]:
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


def verify(graph: dict) -> list[str]:
    """Cheap structural audit: every input link exists, every link endpoint exists.

    Negative node ids are the subgraph I/O proxies ComfyUI uses (-10 = inputs,
    -20 = outputs); they are legitimate endpoints and are skipped.
    """
    problems = []
    ids = {n["id"] for n in graph["nodes"]}
    link_ids = {l["id"] for l in graph["links"]}
    for link in graph["links"]:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source workflow .json")
    ap.add_argument("--out", help="where to write the patched workflow")
    ap.add_argument("--node", default="PatchTuringCutlassAttention", help="MODEL->MODEL node type to inject")
    ap.add_argument("--container", choices=["auto", "top", "subgraph"], default="auto")
    ap.add_argument("--subgraph-index", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    args = ap.parse_args()

    doc = json.loads(Path(args.src).read_text(encoding="utf-8"))
    graph, label = pick_container(doc, args.container, args.subgraph_index)
    print(f"graph: {label}  ({len(graph['nodes'])} nodes, {len(graph['links'])} links)")

    links = find_tail_links(graph)
    if not links:
        sys.exit("error: no MODEL-carrying link into a sampler/guider was found.\n"
                 "       Is this an H3 workflow? Check that the UNET->guider chain is intact.")

    upstream = {(l["origin_id"], l["origin_slot"]) for l in links}
    if len(upstream) > 1:
        detail = "\n".join(f"  node {o} slot {s} -> {sum(1 for l in links if (l['origin_id'], l['origin_slot']) == (o, s))} consumer(s)"
                           for o, s in sorted(upstream))
        sys.exit("error: the sampler/guider models come from several different producers,\n"
                 "       so a single insertion point cannot serve them all:\n" + detail +
                 "\n       Insert the node manually after the merge point, or patch each branch separately.")

    origin_id, origin_slot = upstream.pop()
    nodes = {n["id"]: n for n in graph["nodes"]}
    print(f"insertion point: after node {origin_id} ({nodes[origin_id]['type']}) slot {origin_slot}")
    print(f"will repoint {len(links)} consumer link(s): {[l['id'] for l in links]}")

    if args.dry_run:
        print("dry run, nothing written")
        return 0

    pos = nodes[origin_id].get("pos") or [0, 0]
    size = nodes[origin_id].get("size") or [300, 100]
    new_node = insert_after(graph, origin_id, origin_slot, links, args.node,
                            [pos[0] + size[0] + 70, pos[1] + 20])

    problems = verify(graph)
    if problems:
        print("structural audit FAILED:")
        for p in problems:
            print("  " + p)
        return 1

    out = Path(args.out or args.src)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"inserted node {new_node['id']} ({args.node}) -> {out}")
    print("structural audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
