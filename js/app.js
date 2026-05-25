/* TAMArt project page — interactive explorer.
   Loads precomputed assets (no model inference): per-painting captions, span
   token-activation maps (base64 uint8 on the vision grid), and SAM 3 masks.   */
"use strict";

const CAT_COLORS = {
  CVO:"#1f77b4", ICON:"#9467bd", STYLE:"#2ca02c", AFFECT:"#d62728", META:"#ff7f0e", "?":"#7f7f7f"
};
const CAT_NAME = {
  CVO:"Concrete object", ICON:"Named figure", STYLE:"Style", AFFECT:"Affect", META:"Metadata"
};
const SEG_CATS = new Set(["CVO","ICON"]);

const el = id => document.getElementById(id);
const overlay = el("overlay"), octx = overlay.getContext("2d");
const painting = el("painting");

let INDEX = [];                 // gallery list
let cur = null;                 // current painting data
let sel = -1;                   // selected span index
let samImg = null;              // preloaded SAM mask <img> for selected span
let activeFilter = "ALL";

/* ---------- inferno-ish colormap ---------- */
const CM = [
  [0,[0,0,4]],[0.13,[31,12,72]],[0.25,[85,15,109]],[0.38,[136,34,106]],
  [0.5,[186,54,85]],[0.63,[227,89,51]],[0.75,[249,140,10]],[0.88,[249,201,50]],[1,[252,255,164]]
];
function cmap(t){
  t = t<0?0:t>1?1:t;
  for(let i=1;i<CM.length;i++){
    if(t<=CM[i][0]){
      const [a,ca]=CM[i-1], [b,cb]=CM[i], f=(t-a)/(b-a||1);
      return [ca[0]+(cb[0]-ca[0])*f, ca[1]+(cb[1]-ca[1])*f, ca[2]+(cb[2]-ca[2])*f];
    }
  }
  return CM[CM.length-1][1];
}

/* ---------- helpers ---------- */
function b64ToBytes(s){
  const bin = atob(s), out = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) out[i]=bin.charCodeAt(i);
  return out;
}
function otsu(values){               // values: Uint8ClampedArray of 0..255 samples
  const hist = new Array(256).fill(0);
  for(let i=0;i<values.length;i++) hist[values[i]]++;
  const total = values.length;
  let sum=0; for(let i=0;i<256;i++) sum+=i*hist[i];
  let sumB=0, wB=0, max=-1, thr=0;
  for(let t=0;t<256;t++){
    wB+=hist[t]; if(!wB) continue;
    const wF=total-wB; if(!wF) break;
    sumB+=t*hist[t];
    const mB=sumB/wB, mF=(sum-sumB)/wF;
    const between=wB*wF*(mB-mF)*(mB-mF);
    if(between>max){ max=between; thr=t; }
  }
  return thr;
}
function esc(s){ return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

/* ---------- data loading ---------- */
async function loadIndex(){
  const r = await fetch("assets/data/index.json");
  INDEX = (await r.json()).paintings;
  buildFilters();
}
async function loadPainting(stem){
  const r = await fetch(`assets/data/paintings/${stem}.json`);
  cur = await r.json();
  sel = -1; samImg = null;
  el("np-title").textContent = cur.gt_title || stem;
  el("np-artist").textContent = [cur.gt_artist, cur.year].filter(Boolean).join(" · ");
  renderLegend();
  renderCaption();
  renderMetaCard();
  // load image, then auto-select a representative span
  painting.onload = () => { sizeCanvas(); autoSelect(); };
  painting.src = cur.image;
  if(painting.complete && painting.naturalWidth){ sizeCanvas(); autoSelect(); }
}
function autoSelect(){
  let i = cur.spans.findIndex(s => /village|town|woman|man|figure|boat|tree|house/i.test(s.word) && s.cat==="CVO");
  if(i<0) i = cur.spans.findIndex(s => s.cat==="CVO");
  if(i<0) i = 0;
  selectSpan(i);
}

/* ---------- caption + legend ---------- */
function renderLegend(){
  const present = {};
  cur.spans.forEach(s => present[s.cat]=(present[s.cat]||0)+1);
  el("legend").innerHTML = Object.keys(present).sort().map(c =>
    `<span class="lg"><span class="dot" style="background:${CAT_COLORS[c]}"></span>${CAT_NAME[c]||c} <span style="opacity:.6">(${present[c]})</span></span>`
  ).join("");
}
function renderCaption(){
  // map span-id -> running index in cur.spans (segments use the same ids)
  let html = "";
  for(const seg of cur.segments){
    if(seg.s < 0){ html += `<span class="plain">${esc(seg.t)}</span>`; continue; }
    const cat = (cur.spans[seg.s]||{}).cat || "?";
    html += `<span class="sp" data-s="${seg.s}" style="background:${CAT_COLORS[cat]}">${esc(seg.t)}</span>`;
  }
  const c = el("caption"); c.innerHTML = html;
  c.querySelectorAll(".sp").forEach(n =>
    n.addEventListener("click", () => selectSpan(parseInt(n.dataset.s,10)))
  );
}

function selectSpan(i){
  sel = i;
  const span = cur.spans[i];
  document.querySelectorAll(".caption .sp").forEach(n =>
    n.classList.toggle("active", parseInt(n.dataset.s,10)===i));
  el("span-hint").innerHTML = `Showing <b style="color:${CAT_COLORS[span.cat]}">${esc(span.word)}</b> &mdash; ${CAT_NAME[span.cat]||span.cat}`;

  // SAM availability
  const samCtl = el("ctl-sam"), samToggle = el("t-sam");
  if(span.sam){
    samCtl.classList.remove("disabled"); samToggle.disabled = false;
    el("sam-note").textContent = `SAM 3 found ${span.sam_n} instance${span.sam_n>1?"s":""} for “${span.word}”.`;
    samImg = new Image();
    samImg.onload = renderOverlay;
    samImg.src = `assets/masks/${span.sam}`;
  } else {
    samCtl.classList.add("disabled"); samToggle.checked = false; samToggle.disabled = true;
    samImg = null;
    el("sam-note").textContent = SEG_CATS.has(span.cat)
      ? "SAM 3 returned no detection for this span."
      : "SAM 3 comparison applies to concrete objects and named figures only.";
  }
  renderOverlay();
}

/* ---------- canvas / overlay ---------- */
function sizeCanvas(){
  const w = painting.clientWidth, h = painting.clientHeight;
  if(!w || !h) return;
  overlay.width = w; overlay.height = h;
}
function renderOverlay(){
  octx.clearRect(0,0,overlay.width,overlay.height);
  if(sel<0 || !cur) return;
  const W = overlay.width, H = overlay.height;
  if(!W||!H) return;
  const span = cur.spans[sel];
  const gw = cur.grid_w, gh = cur.grid_h;
  const bytes = b64ToBytes(span.map);
  const alpha = parseInt(el("alpha").value,10)/100;

  // small grayscale grid -> upscale (bilinear) to working canvas
  const grid = document.createElement("canvas"); grid.width=gw; grid.height=gh;
  const gctx = grid.getContext("2d");
  const gimg = gctx.createImageData(gw,gh);
  for(let i=0;i<gw*gh;i++){ const v=bytes[i]; gimg.data[i*4]=v; gimg.data[i*4+1]=v; gimg.data[i*4+2]=v; gimg.data[i*4+3]=255; }
  gctx.putImageData(gimg,0,0);

  const work = document.createElement("canvas"); work.width=W; work.height=H;
  const wctx = work.getContext("2d");
  wctx.imageSmoothingEnabled = true; wctx.imageSmoothingQuality="high";
  wctx.drawImage(grid,0,0,W,H);
  const vals = wctx.getImageData(0,0,W,H).data;   // red channel = upscaled value

  // ----- TAM heatmap -----
  if(el("t-tam").checked){
    const out = octx.createImageData(W,H);
    for(let p=0;p<W*H;p++){
      const v = vals[p*4]/255;
      const [r,g,b] = cmap(v);
      out.data[p*4]=r; out.data[p*4+1]=g; out.data[p*4+2]=b;
      // emphasise hot regions: alpha grows with value
      out.data[p*4+3] = Math.round(255*alpha*Math.pow(v,1.15));
    }
    // putImageData ignores globalAlpha, so composite via temp canvas
    const tmp=document.createElement("canvas"); tmp.width=W; tmp.height=H;
    tmp.getContext("2d").putImageData(out,0,0);
    octx.drawImage(tmp,0,0);
  }

  // ----- Otsu threshold of the TAM map -----
  if(el("t-otsu").checked){
    const samp = new Uint8ClampedArray(W*H);
    for(let p=0;p<W*H;p++) samp[p]=vals[p*4];
    const thr = otsu(samp);
    const out = octx.createImageData(W,H);
    for(let p=0;p<W*H;p++){
      if(samp[p]>=thr && thr>0){
        out.data[p*4]=255; out.data[p*4+1]=138; out.data[p*4+2]=61; out.data[p*4+3]=Math.round(150*alpha);
      }
    }
    const tmp=document.createElement("canvas"); tmp.width=W; tmp.height=H;
    tmp.getContext("2d").putImageData(out,0,0);
    octx.drawImage(tmp,0,0);
  }

  // ----- SAM 3 mask (pre-tinted RGBA) -----
  if(el("t-sam").checked && samImg && samImg.complete){
    octx.globalAlpha = Math.min(1, alpha+0.05);
    octx.drawImage(samImg,0,0,W,H);
    octx.globalAlpha = 1;
  }
}

/* ---------- meta card ---------- */
function renderMetaCard(){
  const m = cur.meta || {}, card = el("meta-card");
  if(!m.title_pred && !m.artist_pred){ card.classList.remove("show"); card.innerHTML=""; return; }
  const v = ok => ok===true ? '<span class="verdict ok">correct</span>'
                : ok===false ? '<span class="verdict no">wrong</span>' : "";
  let rows = `<div class="row"><span class="k">Model wrote</span><span>this is what the caption claimed about the work's identity:</span></div>`;
  if(m.title_pred)  rows += `<div class="row"><span class="k">Predicted title</span><span>“${esc(m.title_pred)}” ${v(m.title_correct)}</span></div>`;
  if(m.artist_pred) rows += `<div class="row"><span class="k">Predicted artist</span><span>${esc(m.artist_pred)} ${v(m.artist_correct)}</span></div>`;
  rows += `<div class="row"><span class="k">Ground truth</span><span>${esc(cur.gt_title||"?")} — ${esc(cur.gt_artist||"?")}</span></div>`;
  card.innerHTML = rows; card.classList.add("show");
}

/* ---------- picker ---------- */
function buildFilters(){
  const cats = ["ALL","ICON","CVO","STYLE","AFFECT","META"];
  el("filters").innerHTML = cats.map(c =>
    `<button class="f${c==="ALL"?" on":""}" data-f="${c}">${c==="ALL"?"All":c}</button>`).join("");
  el("filters").querySelectorAll(".f").forEach(b =>
    b.addEventListener("click", () => {
      activeFilter=b.dataset.f;
      el("filters").querySelectorAll(".f").forEach(x=>x.classList.toggle("on",x===b));
      renderGrid();
    }));
}
function renderGrid(){
  const q = el("search").value.trim().toLowerCase();
  const items = INDEX.filter(p => {
    if(activeFilter!=="ALL" && !(p.cats||[]).includes(activeFilter)) return false;
    if(q && !((p.title||"").toLowerCase().includes(q) || (p.artist||"").toLowerCase().includes(q))) return false;
    return true;
  });
  el("grid").innerHTML = items.map(p => `
    <div class="card" data-stem="${p.stem}">
      <img loading="lazy" src="${p.thumb}" alt="${esc(p.title||"")}" />
      <div class="cap">
        <b>${esc(p.title||p.stem)}</b>
        <span>${esc(p.artist||"")}${p.year?" · "+p.year:""}</span>
        ${p.n_sam?`<div class="badge">${p.n_sam} SAM masks</div>`:""}
      </div>
    </div>`).join("") || `<p style="color:var(--mut)">No matches.</p>`;
  el("grid").querySelectorAll(".card").forEach(c =>
    c.addEventListener("click", () => { closePicker(); loadPainting(c.dataset.stem); }));
}
function openPicker(){ el("picker").classList.remove("hidden"); renderGrid(); el("search").focus(); }
function closePicker(){ el("picker").classList.add("hidden"); }

/* ---------- wire up ---------- */
["t-tam","t-otsu","t-sam"].forEach(id => el(id).addEventListener("change", renderOverlay));
el("alpha").addEventListener("input", renderOverlay);
el("pick-btn").addEventListener("click", openPicker);
el("picker-close").addEventListener("click", closePicker);
el("picker").addEventListener("click", e => { if(e.target===el("picker")) closePicker(); });
el("search").addEventListener("input", renderGrid);
document.addEventListener("keydown", e => { if(e.key==="Escape") closePicker(); });
let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt=setTimeout(()=>{ sizeCanvas(); renderOverlay(); },120); });

/* ---------- boot ---------- */
(async function(){
  await loadIndex();
  const start = INDEX.find(p => p.stem.includes("the-starry-night")) || INDEX[0];
  if(start) loadPainting(start.stem);
})();
