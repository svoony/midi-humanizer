"""
Visual eval viewer: pick a held-out TEST piece and eyeball the model's
velocity and rubato (onset-timing) curves against the ground-truth human
performance.

The test split is never used in training or early stopping (train.py loads
only train+validation), so these are genuinely unseen pieces. For each piece
the ground truth comes from original.midi; the prediction is the model's
deadpan (prior-z) render via infer.predict_raw - i.e. exactly what the player
would produce. Raw per-note arrays are sent to the browser, which smooths and
plots them, so the smoothing window is interactive (the phrase-vs-local
distinction the correlation eval surfaced).

Run:  python eval_ui.py     then open  http://127.0.0.1:5001
"""
import csv
import os

import numpy as np
from flask import Flask, Response, jsonify, request

import infer
import legacy
from note_dataset import (
    MAESTRO_CSV_PATH, _load_piece_arrays, era_for_composer, load_style_stats,
)

STYLE_MEAN, STYLE_STD = load_style_stats()

ROOT = "paired_data"
app = Flask(__name__)
_curve_cache = {}


def build_test_index():
    items = []
    with open(MAESTRO_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            year, filename = row["midi_filename"].split("/", 1)
            piece_name = os.path.splitext(filename)[0]
            piece_dir = os.path.join(ROOT, "test", year, piece_name)
            if not os.path.isdir(piece_dir):
                continue
            items.append({
                "dir": piece_dir,
                "label": f"{row['canonical_composer']} — {row['canonical_title']}",
                "year": year,
                "era": era_for_composer(row["canonical_composer"]),
            })
    items.sort(key=lambda d: d["label"])
    return items


TEST_INDEX = build_test_index()
DIR_TO_META = {d["dir"]: d for d in TEST_INDEX}


def compute_curves(piece_dir, era_id):
    if piece_dir in _curve_cache:
        return _curve_cache[piece_dir]

    src = os.path.join(piece_dir, "original.midi")
    p = _load_piece_arrays(piece_dir)
    vel_gt = p["velocities"].astype(np.float64) / 127.0

    # gen-3: condition on this performance's own style; rubato = local-tempo
    # log-ratio (its representation), GT = cached tempo_dev
    style_z = (p["style"].astype(np.float32) - STYLE_MEAN) / STYLE_STD
    raw = infer.predict_raw(src, era_id, style_z=style_z)
    rub3_gt = p["tempo_dev"]

    # gen-2: its own un-folded grid + bounded onset-offset representation
    g2 = legacy.gen2_predict(src, era_id)
    rub2_gt_full = p["starts"] - g2["q_unfolded"][: len(p["starts"])]

    n = min(len(raw["q_starts"]), len(rub3_gt), len(g2["q_unfolded"]), len(rub2_gt_full))
    data = {
        "label": DIR_TO_META[piece_dir]["label"],
        "n": int(n),
        "x": np.round(raw["q_starts"][:n], 3).tolist(),
        # velocity (same 0-1 target in both gens -> directly overlaid)
        "vel_gt": np.round(vel_gt[:n], 4).tolist(),
        "vel_gen2": np.round(g2["velocity_pred"][:n], 4).tolist(),
        "vel_gen3": np.round(raw["velocity"][:n], 4).tolist(),
        # gen-3 rubato: local-tempo log-ratio
        "rub3_gt": np.round(rub3_gt[:n], 4).tolist(),
        "rub3_pred": np.round(raw["log_ioi_ratio"][:n], 4).tolist(),
        # gen-2 rubato: onset offset vs nearest grid, in seconds (bounded)
        "rub2_gt": np.round(rub2_gt_full[:n], 4).tolist(),
        "rub2_pred": np.round(g2["timing_offset_pred"][:n], 4).tolist(),
    }
    _curve_cache[piece_dir] = data
    return data


@app.route("/api/pieces")
def api_pieces():
    return jsonify([{"dir": d["dir"], "label": d["label"], "year": d["year"]} for d in TEST_INDEX])


@app.route("/api/piece")
def api_piece():
    piece_dir = request.args.get("dir")
    meta = DIR_TO_META.get(piece_dir)
    if meta is None:
        return jsonify({"error": "unknown piece"}), 404
    return jsonify(compute_curves(piece_dir, meta["era"]))


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>CVAE eval viewer</title>
<style>
  :root { --gt:#1f77b4; --pred:#e8590c; --gen2:#8a8a8a; --line:#ddd; --fg:#222; --muted:#777; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; color:var(--fg); display:flex; height:100vh; }
  #side { width:320px; border-right:1px solid var(--line); display:flex; flex-direction:column; }
  #side h1 { font-size:14px; margin:0; padding:12px; border-bottom:1px solid var(--line); }
  #search { margin:8px; padding:6px; font-size:13px; }
  #list { overflow-y:auto; flex:1; }
  .item { padding:8px 12px; font-size:12.5px; cursor:pointer; border-bottom:1px solid #f2f2f2; }
  .item:hover { background:#f5f7fa; }
  .item.active { background:#e7f0fa; font-weight:600; }
  .item .yr { color:var(--muted); font-weight:400; }
  #main { flex:1; padding:16px 24px; overflow-y:auto; }
  #title { font-size:16px; margin:0 0 4px; }
  #meta { color:var(--muted); font-size:12.5px; margin-bottom:12px; }
  #ctrl { display:flex; align-items:center; gap:10px; font-size:13px; margin-bottom:8px; }
  .legend span { display:inline-flex; align-items:center; gap:5px; margin-right:14px; font-size:12.5px; }
  .swatch { width:14px; height:3px; display:inline-block; border-radius:2px; }
  .chartwrap { margin-bottom:18px; }
  .chartwrap h3 { font-size:13px; margin:0 0 2px; }
  svg { width:100%; height:240px; display:block; }
  .axis { stroke:#bbb; stroke-width:1; }
  .grid { stroke:#eee; stroke-width:1; }
  .tick { fill:var(--muted); font-size:10px; }
  #empty { color:var(--muted); margin-top:40px; }
</style></head>
<body>
  <div id="side">
    <h1>Held-out TEST pieces</h1>
    <input id="search" placeholder="filter by composer / title...">
    <div id="list"></div>
  </div>
  <div id="main">
    <h2 id="title">Select a test piece</h2>
    <div id="meta"></div>
    <div id="ctrl" style="display:none">
      <label>Smoothing window: <b id="swval">8</b> notes</label>
      <input id="smooth" type="range" min="1" max="64" value="8" style="width:220px">
      <span class="legend"><span><span class="swatch" style="background:var(--gt)"></span>Ground truth</span>
      <span><span class="swatch" style="background:var(--gen2)"></span>gen-2</span>
      <span><span class="swatch" style="background:var(--pred)"></span>gen-3</span></span>
    </div>
    <div id="charts"></div>
    <div id="empty">Pick a piece on the left to compare the human performance against the model.</div>
  </div>

<script>
let CUR = null;
const $ = s => document.querySelector(s);

function smooth(a, w){
  if(w<=1) return a.slice();
  const out=new Array(a.length), h=Math.floor(w/2);
  for(let i=0;i<a.length;i++){let s=0,c=0;
    for(let j=Math.max(0,i-h);j<=Math.min(a.length-1,i+h);j++){s+=a[j];c++;}
    out[i]=s/c;}
  return out;
}

function path(x, y, x0, x1, y0, y1, W, H, pad){
  const sx = v => pad.l + (v-x0)/((x1-x0)||1)*(W-pad.l-pad.r);
  const sy = v => H-pad.b - (v-y0)/((y1-y0)||1)*(H-pad.t-pad.b);
  let d="";
  for(let i=0;i<x.length;i++){ d += (i?"L":"M") + sx(x[i]).toFixed(1) + " " + sy(y[i]).toFixed(1) + " "; }
  return d;
}

function chart(title, x, series, unit, zeroLine){
  const W = $("#charts").clientWidth || 800, H = 220, pad={l:52,r:16,t:10,b:24};
  const allY = [].concat(...series.map(s=>s.y));
  let y0 = Math.min(...allY), y1 = Math.max(...allY);
  if(zeroLine){ const m=Math.max(Math.abs(y0),Math.abs(y1)); y0=-m; y1=m; }
  const padY=(y1-y0)*0.08||1; y0-=padY; y1+=padY;
  const x0=x[0], x1=x[x.length-1];
  const sx = v => pad.l + (v-x0)/((x1-x0)||1)*(W-pad.l-pad.r);
  const sy = v => H-pad.b - (v-y0)/((y1-y0)||1)*(H-pad.t-pad.b);
  let g="";
  for(let k=0;k<=4;k++){ const yv=y0+(y1-y0)*k/4, py=sy(yv);
    g+=`<line class="grid" x1="${pad.l}" y1="${py}" x2="${W-pad.r}" y2="${py}"/>`;
    g+=`<text class="tick" x="${pad.l-6}" y="${py+3}" text-anchor="end">${yv.toFixed(unit==='ms'?0:2)}</text>`; }
  for(let k=0;k<=5;k++){ const xv=x0+(x1-x0)*k/5, px=sx(xv);
    g+=`<text class="tick" x="${px}" y="${H-8}" text-anchor="middle">${xv.toFixed(0)}s</text>`; }
  if(zeroLine){ g+=`<line class="axis" x1="${pad.l}" y1="${sy(0)}" x2="${W-pad.r}" y2="${sy(0)}"/>`; }
  for(const s of series){
    g+=`<path d="${path(x,s.y,x0,x1,y0,y1,W,H,pad)}" fill="none" stroke="${s.color}" stroke-width="1.4"/>`;
  }
  return `<div class="chartwrap"><h3>${title} <span style="color:#999;font-weight:400">(${unit})</span></h3>`+
         `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${g}</svg></div>`;
}

const GT='var(--gt)', G2='var(--gen2)', G3='var(--pred)';
function render(){
  if(!CUR) return;
  const w = +$("#smooth").value; $("#swval").textContent = w;
  const x = CUR.x, S = a => smooth(a, w);
  $("#charts").innerHTML =
    chart("Velocity", x, [
      {y:S(CUR.vel_gt), color:GT}, {y:S(CUR.vel_gen2), color:G2}, {y:S(CUR.vel_gen3), color:G3}
    ], "0–1", false) +
    chart("gen-3 rubato — local tempo (+ slower / − faster); GT has phrase arcs", x, [
      {y:S(CUR.rub3_gt), color:GT}, {y:S(CUR.rub3_pred), color:G3}
    ], "log-ratio", true) +
    chart("gen-2 rubato — onset offset vs grid; GT is bounded ±½ step (no phrase arc)", x, [
      {y:S(CUR.rub2_gt).map(v=>v*1000), color:GT}, {y:S(CUR.rub2_pred).map(v=>v*1000), color:G2}
    ], "ms", true);
}

async function load(dir, el){
  document.querySelectorAll(".item").forEach(n=>n.classList.remove("active"));
  el.classList.add("active");
  $("#title").textContent = "Loading…"; $("#meta").textContent=""; $("#empty").style.display="none";
  const r = await fetch("/api/piece?dir="+encodeURIComponent(dir));
  CUR = await r.json();
  $("#title").textContent = CUR.label;
  $("#meta").textContent = CUR.n + " notes · ground truth vs gen-2 vs gen-3 · each rubato shown in its own representation";
  $("#ctrl").style.display="flex";
  render();
}

async function init(){
  const pieces = await (await fetch("/api/pieces")).json();
  const list = $("#list");
  function draw(filter){
    list.innerHTML="";
    pieces.filter(p=>p.label.toLowerCase().includes(filter)).forEach(p=>{
      const d=document.createElement("div"); d.className="item";
      d.innerHTML = p.label + ' <span class="yr">('+p.year+')</span>';
      d.onclick=()=>load(p.dir, d); list.appendChild(d);
    });
  }
  draw("");
  $("#search").addEventListener("input", e=>draw(e.target.value.toLowerCase()));
  $("#smooth").addEventListener("input", render);
  window.addEventListener("resize", render);
}
init();
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"test pieces: {len(TEST_INDEX)}")
    app.run(host="127.0.0.1", port=5001, debug=False)
