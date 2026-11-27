"""Human evaluation preparation, pooling, annotation and reporting."""
from __future__ import annotations

import argparse, hashlib, json, os, tempfile, threading, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image

from .framing import prepare_image
from .query import _candidate_preview
from .retrieval import Library, QueryObject, rank
from .retrieval_config import load_retrieval_config

_LABEL_LOCK = threading.RLock()

def _read_json(path, default):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return default

def _atomic_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f: json.dump(value, f, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        try: os.unlink(name)
        except FileNotFoundError: pass

def _hash_inputs(image_bytes, x, y, mask_index):
    h = hashlib.sha256(); h.update(image_bytes); h.update(repr((float(x), float(y), int(mask_index))).encode()); return h.hexdigest()

def prepare_queries(spec, out, config=None, device="cpu", segmenter=None):
    spec = Path(spec); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    import tomllib
    # Query specs are hand-written TOML.  Accept JSON too for small fixtures.
    try:
        with spec.open("rb") as f:
            raw = tomllib.load(f)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        raw = json.loads(spec.read_text())
    queries = raw.get("query", [])
    old = {q["id"]: q for q in _read_json(out / "queries.json", {}).get("queries", [])}
    if segmenter is None:
        from .models import SamPointSegmenter
        cache_dir = None
        try:
            from .config import load_paths
            cache_dir = load_paths(config).hf_cache
        except FileNotFoundError:
            pass
        segmenter = SamPointSegmenter(device=device, cache_dir=cache_dir)
    manifests = []
    for q in queries:
        qid = str(q["id"]); image_path = Path(q["image"])
        if not image_path.is_absolute(): image_path = spec.parent / image_path
        image_bytes = image_path.read_bytes(); x, y = float(q["x"]), float(q["y"]); mi = int(q.get("mask_index", 0))
        digest = _hash_inputs(image_bytes, x, y, mi); qdir = out / "queries" / qid; qdir.mkdir(parents=True, exist_ok=True)
        if old.get(qid, {}).get("input_hash") == digest and (qdir / "prepared.png").is_file() and (qdir / "mask.png").is_file():
            manifests.append(old[qid]); continue
        image = np.asarray(Image.open(image_path).convert("RGB")); prepared, transform = prepare_image(image, long_side=640)
        px, py = transform.to_prepared(x, y); detections = segmenter.masks_at_point(prepared, px, py)
        if mi < 0 or mi >= len(detections): raise IndexError(f"mask index {mi} out of range (0..{len(detections)-1}) for query {qid}")
        masks = [np.asarray(getattr(d, "mask", d["mask"] if isinstance(d, dict) else d), dtype=bool) for d in detections]
        Image.fromarray(np.asarray(prepared, dtype=np.uint8)).save(qdir / "prepared.png")
        Image.fromarray((masks[mi].astype(np.uint8) * 255)).save(qdir / "mask.png")
        _candidate_preview(np.asarray(prepared), detections, qdir / "candidates.jpg")
        manifest = {"id": qid, "group": q.get("group", ""), "note": q.get("note", ""), "exclude_clips": list(q.get("exclude_clips", [])), "image": str(image_path), "x": x, "y": y, "prepared_x": float(px), "prepared_y": float(py), "mask_index": mi, "num_candidates": len(masks), "area_frac": float(masks[mi].mean()), "input_hash": digest}
        manifests.append(manifest)
    _atomic_json(out / "queries.json", {"queries": manifests})
    return manifests

def _panel(library, prepared, mask, result):
    try:
        from .render import result_panel
        return result_panel(library, prepared, mask, result)
    except ImportError:
        from .app import _result_gallery_item
        return _result_gallery_item(type("E", (), {"library": library})(), prepared, mask, result, 1)[0]

def pool_results(out, config=None, library_root=None, retrieval_config=None, k=10, modes=("soft", "hard"), preload=False, seed=0):
    out = Path(out); qman = _read_json(out / "queries.json", {}); queries = qman.get("queries", [])
    if library_root is None:
        from .config import load_paths
        library_root = load_paths(config).library_root
    lib = Library.load(library_root, preload=preload); rc = load_retrieval_config(retrieval_config)
    rankings = {}; items_manifest = {}
    for q in queries:
        qid = q["id"]; qdir = out / "queries" / qid; prepared = np.asarray(Image.open(qdir / "prepared.png").convert("RGB")); mask = np.asarray(Image.open(qdir / "mask.png").convert("L")) > 0
        query = QueryObject.from_mask(prepared, mask, None); rankings[qid] = {}; pooled = {}
        excluded = set(q.get("exclude_clips", []))
        for mode in modes:
            started = time.perf_counter(); results = rank(lib, query, mode, semantic_weight=0.0, k=max(int(k), len(lib.objects)), params=rc)
            filtered = [r for r in results if r.clip_id not in excluded][:int(k)]
            rankings[qid][mode] = {"results": [r.to_dict() for r in filtered], "timings": dict(getattr(results, "timings", {})), "elapsed_s": time.perf_counter() - started}
            for r in filtered: pooled[(r.clip_id, int(r.shot_no), int(r.object_id))] = r
        rng = np.random.default_rng(int(seed)); keys = list(pooled); rng.shuffle(keys); entries = []
        for idx, key in enumerate(keys):
            r = pooled[key]; item_id = hashlib.sha1((qid + "|" + "|".join(map(str, key))).encode()).hexdigest()[:16]
            ppath = out / "panels" / qid / f"{item_id}.jpg"; ppath.parent.mkdir(parents=True, exist_ok=True)
            if not ppath.is_file(): Image.fromarray(_panel(lib, prepared, mask, r)).save(ppath, quality=92)
            entries.append({"item_id": item_id, "panel": str(ppath.relative_to(out)), "clip_id": r.clip_id, "shot_no": int(r.shot_no), "start_s": r.start_s, "end_s": r.end_s, "keyframe_id": r.keyframe_id, "t_s": r.t_s})
        items_manifest[qid] = entries
    _atomic_json(out / "rankings.json", rankings); _atomic_json(out / "items.json", {"queries": items_manifest}); return rankings, items_manifest

def load_labels(out): return _read_json(Path(out) / "labels.json", {})

def save_labels(out, query_id, usable, judged, note=""):
    path = Path(out) / "labels.json"
    with _LABEL_LOCK:
        labels = _read_json(path, {}); labels[str(query_id)] = {"usable": list(usable), "judged": list(judged), "note": str(note), "saved_at": datetime.now(timezone.utc).isoformat()}; _atomic_json(path, labels)
    return labels[str(query_id)]

def build_annotation_app(out, config=None, host="127.0.0.1", port=7870, share=False):
    os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(tempfile.gettempdir()) / f"gradio-{os.getuid()}"))
    import gradio as gr
    out = Path(out); queries = _read_json(out / "queries.json", {}).get("queries", []); items = _read_json(out / "items.json", {}).get("queries", {}); labels = load_labels(out)
    state = {"index": 0}
    with gr.Blocks(title="Human evaluation") as app:
        header = gr.Markdown(); note = gr.Markdown(); image = gr.Image(interactive=False); gallery = gr.Gallery(); checks = gr.CheckboxGroup(label="Usable for a match cut"); textbox = gr.Textbox(label="Note"); jump = gr.Dropdown(choices=[q["id"] for q in queries], label="Query"); save = gr.Button("Save & next"); prev = gr.Button("Previous"); video = gr.Video(interactive=False); message = gr.Markdown()
        def show(i):
            i = max(0, min(int(i), len(queries)-1)); state["index"] = i; q = queries[i]; its = items.get(q["id"], []); lab = labels.get(q["id"], {}); header_v = f"query {i+1} / {len(queries)} · {len(set(lab.get('judged', [])) & {x['item_id'] for x in its})} labeled"
            prepared_path = out / "queries" / q["id"] / "prepared.png"
            prepared = np.asarray(Image.open(prepared_path).convert("RGB"))
            mask_path = out / "queries" / q["id"] / "mask.png"
            if mask_path.is_file():
                import cv2
                mask = np.asarray(Image.open(mask_path).convert("L")) > 0
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.polylines(prepared, contours, True, (255, 70, 40), 3)
            return header_v, f"**{q.get('note','')}**  ·  {q.get('group','')}", prepared, [(str(out / x["panel"]), f"#{j+1}") for j,x in enumerate(its)], gr.update(choices=[f"#{j+1}" for j in range(len(its))], value=[f"#{j+1}" for j,x in enumerate(its) if x["item_id"] in lab.get("usable", [])]), lab.get("note", ""), q["id"]
        def do_save(chosen, text, qid):
            its = items.get(qid, []); usable = [its[int(str(c)[1:]) - 1]["item_id"] for c in (chosen or [])]; save_labels(out, qid, usable, [x["item_id"] for x in its], text); labels.update(load_labels(out)); return show(min(state["index"] + 1, len(queries)-1))
        def on_gallery_select(evt: gr.SelectData):
            if evt is None or evt.index is None or not queries:
                return None, ""
            idx = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
            q = queries[state["index"]]; its = items.get(q["id"], [])
            try:
                item = its[int(idx)]
            except (IndexError, TypeError, ValueError):
                return None, ""
            try:
                paths = __import__("matchcut_retriever.config", fromlist=["load_paths"]).load_paths(config)
                source = Path(paths.video_root) / f"{item['clip_id']}.mp4"
                if not source.is_file() or item.get("start_s") is None or item.get("end_s") is None:
                    return None, "Video is unavailable for this result."
                from .app import cut_shot_clip
                target = out / "clips" / f"{item['item_id']}.mp4"
                if not target.is_file() and not cut_shot_clip(source, item["start_s"], item["end_s"], target):
                    return None, "Unable to create the shot clip."
                return str(target), ""
            except (FileNotFoundError, KeyError, OSError) as exc:
                return None, f"Video is unavailable: {exc}"
        on_gallery_select.__annotations__["evt"] = gr.SelectData
        app.load(lambda: show(0), outputs=[header,note,image,gallery,checks,textbox,jump]); jump.change(lambda q: show([x["id"] for x in queries].index(q)), jump, [header,note,image,gallery,checks,textbox,jump]); save.click(do_save, [checks,textbox,jump], [header,note,image,gallery,checks,textbox,jump]); prev.click(lambda: show(state["index"]-1), outputs=[header,note,image,gallery,checks,textbox,jump]); gallery.select(on_gallery_select, inputs=[], outputs=[video, message])
    app.matchcut_allowed_paths = [str(out)]; app.human_eval_out = out; return app

def report_results(out, k=10):
    out = Path(out); rankings = _read_json(out / "rankings.json", {}); qman = {q["id"]: q for q in _read_json(out / "queries.json", {}).get("queries", [])}; labels = load_labels(out); modes = sorted({m for q in rankings.values() for m in q}); rows=[]; summary={}
    items_by_query = _read_json(out / "items.json", {}).get("queries", {})
    item_lookup = {(qid, (x.get("clip_id"), int(x.get("shot_no", 0)), x.get("object_id"))): x.get("item_id") for qid, arr in items_by_query.items() for x in arr}
    def result_item_id(qid, result):
        key = (qid, (result.get("clip_id"), int(result.get("shot_no", 0)), result.get("object_id")))
        fallback = hashlib.sha1((qid + "|" + "|".join(str(result.get(x)) for x in ("clip_id", "shot_no", "object_id"))).encode()).hexdigest()[:16]
        return item_lookup.get(key) or fallback
    for mode in modes:
        vals=[]; hits=[]; returned=[]
        for qid, modeset in rankings.items():
            if qid not in labels: continue
            usable = set(labels[qid].get("usable", []))
            rs = modeset.get(mode, {}).get("results", [])[:k]
            ids = []
            for r in rs:
                ids.append(result_item_id(qid, r))
            n = sum(i in usable for i in ids)
            vals.append(n / k); hits.append(bool(n)); returned.append(len(rs))
            rows.append({"query": qid, "group": qman.get(qid, {}).get("group", ""), "mode": mode, "usable": n, "returned": len(rs)})
        summary[mode]={"P@K": float(np.mean(vals)) if vals else 0.0, "precision_returned": float(sum(r["usable"] for r in rows if r["mode"]==mode)/max(1,sum(r["returned"] for r in rows if r["mode"]==mode))), "mean_returned": float(np.mean(returned)) if returned else 0.0, "Hit@K": float(np.mean(hits)) if hits else 0.0}
    unjudged=[]; its=_read_json(out / "items.json", {}).get("queries", {})
    for qid, arr in its.items():
        if qid in labels: unjudged.extend([x["item_id"] for x in arr if x["item_id"] not in labels[qid].get("judged", [])])
    groups = {}
    for group in sorted({r.get("group", "") for r in rows}):
        groups[group] = {}
        for mode in modes:
            gr = [r for r in rows if r.get("group", "") == group and r["mode"] == mode]
            groups[group][mode] = {"P@K": float(np.mean([r["usable"] / k for r in gr])) if gr else 0.0, "Hit@K": float(np.mean([r["usable"] > 0 for r in gr])) if gr else 0.0, "mean_returned": float(np.mean([r["returned"] for r in gr])) if gr else 0.0}
    overlap = {}
    for qid, modeset in rankings.items():
        if qid not in labels: continue
        sets = []
        for mode in modes:
            sets.append({result_item_id(qid, r) for r in modeset.get(mode, {}).get("results", [])})
        overlap[qid] = sorted(x for x in (set.intersection(*sets) if sets else set()) if x)
    payload={"k":k,"modes":summary,"groups":groups,"per_query":rows,"overlap":overlap,"unjudged":unjudged}; _atomic_json(out / "report.json", payload); md="# Human evaluation report\n\n"+"\n".join(f"- **{m}**: P@{k}={v['P@K']:.3f}, Hit@{k}={v['Hit@K']:.3f}, precision returned={v['precision_returned']:.3f}, mean returned={v['mean_returned']:.2f}" for m,v in summary.items()); (out/"report.md").write_text(md); print(md); return payload

prepare = prepare_queries
pool = pool_results
annotate = build_annotation_app
report = report_results

def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("prepare"); a.add_argument("--spec",required=True); a.add_argument("--out",required=True); a.add_argument("--config"); a.add_argument("--device",default="cpu")
    a=sub.add_parser("pool"); a.add_argument("--out",required=True); a.add_argument("--config"); a.add_argument("--library-root"); a.add_argument("--retrieval-config"); a.add_argument("--k",type=int,default=10); a.add_argument("--modes",default="soft,hard"); a.add_argument("--preload",action="store_true"); a.add_argument("--seed",type=int,default=0)
    a=sub.add_parser("annotate"); a.add_argument("--out",required=True); a.add_argument("--config"); a.add_argument("--host",default="127.0.0.1"); a.add_argument("--port",type=int,default=7870); a.add_argument("--share",action="store_true")
    a=sub.add_parser("report"); a.add_argument("--out",required=True); a.add_argument("--k",type=int,default=10)
    args=p.parse_args(argv)
    if args.cmd=="prepare": prepare_queries(args.spec,args.out,args.config,args.device)
    elif args.cmd=="pool": pool_results(args.out,args.config,args.library_root,args.retrieval_config,args.k,tuple(m.strip() for m in args.modes.split(",") if m.strip()),args.preload,args.seed)
    elif args.cmd == "annotate":
        app = build_annotation_app(args.out, args.config)
        app.queue().launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=[str(Path(args.out))])
    else: report_results(args.out,args.k)
    return 0
if __name__ == "__main__": raise SystemExit(main())
