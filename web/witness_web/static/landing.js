"use strict";
// Witness landing. Every overlay below is an illustration on public still frames, not model output.
(() => {
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (s, r = document) => r.querySelector(s);
const onScreen = (node, fn, threshold = .35) => { if (!("IntersectionObserver" in window)) return fn(); const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); } }, {threshold}); io.observe(node); };

// Reveal on scroll.
const revealables = document.querySelectorAll("[data-reveal]");
if (reduced || !("IntersectionObserver" in window)) revealables.forEach((n) => n.classList.add("in"));
else { const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), {rootMargin: "0px 0px -10% 0px"}); revealables.forEach((n) => io.observe(n)); }

// 02 · Claim format, one modality highlighted at a time. Static with reduced motion. ---
const fam = $(".fam-stage");
if (fam) {
  const chips = [...fam.querySelectorAll(".chip")], items = [...document.querySelectorAll(".fam-list li")];
  if (reduced) chips.forEach((c) => c.classList.add("on"));
  else {
    const order = ["visual", "speech", "text", "sound"]; let k = 0, t;
    const cycle = () => { const key = order[k % order.length]; chips.forEach((c) => c.classList.toggle("on", c.classList.contains(key))); items.forEach((li) => li.classList.toggle("on", li.dataset.fam === key)); k++; t = setTimeout(cycle, 1900); };
    onScreen(fam, cycle);
    document.addEventListener("visibilitychange", () => { clearTimeout(t); if (!document.hidden && k) t = setTimeout(cycle, 400); });
  }
}
})();

// Hero finder: example claims marked on still frames (illustrative, no timing). -----
(() => {
  const finder = document.getElementById("finder"); if (!finder) return;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const img = finder.querySelector(".finder-img"), canvas = finder.querySelector(".finder-canvas"), ctx = canvas.getContext("2d"),
    thumbs = [...finder.querySelectorAll(".filmstrip img")], line = finder.querySelector(".finder-line"), count = finder.querySelector(".finder-count"),
    claim = finder.querySelector(".finder-claim");
  // [modality, example claim, x, y, w, h] in fractions of the frame.
  const STILLS = [
    {src: "/static/media/kitchen.jpg", claims: [["visual", "a slice of berry cake on a plate", .55, .26, .36, .52], ["text", "\"BOOK\" printed on the cover", .01, .2, .26, .18], ["visual", "berries and cream on top", .62, .25, .17, .22]]},
    {src: "/static/media/woodpecker.jpg", claims: [["visual", "a woodpecker clings to the trunk", .53, .58, .22, .4], ["visual", "a nest hole in the bark", .34, .36, .14, .3]]},
    {src: "/static/media/workshop.jpg", claims: [["visual", "a person at a band saw", .15, .03, .25, .9], ["visual", "hands guide a board to the blade", .39, .32, .16, .16]]},
    {src: "/static/media/nature.jpg", claims: [["visual", "a bee on a flower", .2, .32, .3, .5], ["visual", "a pink flower spike in close-up", .45, .25, .44, .55]]},
  ];
  let s = 0, c = 0, box = null, t0 = 0, raf, timer, running = false;
  const resize = () => { const r = canvas.getBoundingClientRect(); canvas.width = Math.round(r.width * devicePixelRatio); canvas.height = Math.round(r.height * devicePixelRatio); };
  const render = (now, still) => {
    const w = canvas.width, h = canvas.height; ctx.clearRect(0, 0, w, h); if (!box) return;
    const p = still ? 1 : Math.min(1, (now - t0) / 550), e = 1 - Math.pow(1 - p, 3);
    const x = (box.x + (0.5 - box.x) * (1 - e)) * w, y = (box.y + (0.5 - box.y) * (1 - e)) * h, bw = (box.w + (1 - box.w) * (1 - e)) * w, bh = (box.h + (1 - box.h) * (1 - e)) * h;
    ctx.strokeStyle = "#cafc70"; ctx.lineWidth = Math.max(1, devicePixelRatio); ctx.strokeRect(x, y, bw, bh);
    const L = 10 * devicePixelRatio; ctx.lineWidth *= 2.2; for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + bw, y, -1, 1], [x, y + bh, 1, -1], [x + bw, y + bh, -1, -1]]) { ctx.beginPath(); ctx.moveTo(cx, cy + dy * L); ctx.lineTo(cx, cy); ctx.lineTo(cx + dx * L, cy); ctx.stroke(); }
    if (p >= 1) { ctx.font = `${10 * devicePixelRatio}px ui-monospace, Menlo, monospace`; const label = ` ${box.label} `, tw = ctx.measureText(label).width, lh = 16 * devicePixelRatio;
      const ly = y + bh + lh <= h ? y + bh : Math.max(0, y - lh); ctx.fillStyle = "#cafc70"; ctx.fillRect(x, ly, tw, lh); ctx.fillStyle = "#0a0c0a"; ctx.fillText(label, x, ly + 11.5 * devicePixelRatio); }
    // scan line
    if (!still) { ctx.fillStyle = "#cafc7040"; ctx.fillRect(x, y + bh * ((now / 12) % 100) / 100, bw, 2 * devicePixelRatio); }
  };
  const show = () => {
    const S = STILLS[s], [modality, text, x, y, bw, bh] = S.claims[c];
    if (!img.src.endsWith(S.src)) img.src = S.src;
    box = {label: modality, x, y, w: bw, h: bh}; t0 = performance.now();
    line.textContent = `example · ${modality} · ${text}`; count.textContent = `still ${s + 1} / ${STILLS.length}`;
    claim.textContent = text.charAt(0).toUpperCase() + text.slice(1) + ".";
    thumbs.forEach((t, i) => t.classList.toggle("hot", i === s));
  };
  const advance = () => { c++; if (c >= STILLS[s].claims.length) { c = 0; s = (s + 1) % STILLS.length; } show(); timer = setTimeout(advance, 2200); };
  const frame = (now) => { render(now, false); raf = requestAnimationFrame(frame); };
  const start = () => { if (running) return; running = true; raf = requestAnimationFrame(frame); timer = setTimeout(advance, 2200); };
  const stop = () => { running = false; clearTimeout(timer); cancelAnimationFrame(raf); };
  resize(); show();
  if (reduced) { render(0, true); addEventListener("resize", () => { resize(); render(0, true); }); return; }
  addEventListener("resize", resize);
  let visible = !("IntersectionObserver" in window);
  if (visible) start(); else new IntersectionObserver((es) => { visible = es.some((e) => e.isIntersecting); if (visible) start(); else stop(); }, {threshold: .2}).observe(finder);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); else if (visible) start(); });
})();
