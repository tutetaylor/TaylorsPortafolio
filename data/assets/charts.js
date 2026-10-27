/* charts.js — mini librería de gráficos SVG sin dependencias (Mateo Taylor · Codexa) */
(function (g) {
  const NS = 'http://www.w3.org/2000/svg';
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const PAL = () => [css('--c1') || '#c5f23a', css('--c2') || '#5fd4c4', css('--c3') || '#f2c14e', css('--c4') || '#e86f5a', css('--c5') || '#9b8cf2', css('--c6') || '#6c7465'];
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const fmtNum = (v, f) => typeof f === 'function' ? f(v) : (Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : Math.abs(v) >= 1e4 ? (v / 1e3).toFixed(0) + 'k' : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(1) + 'k' : (Math.round(v * 100) / 100).toString());

  let tip;
  function tooltip() {
    if (tip) return tip;
    tip = document.createElement('div'); tip.className = 'dc-tip'; document.body.appendChild(tip);
    return tip;
  }
  function showTip(html, x, y) {
    const t = tooltip(); t.innerHTML = html; t.style.opacity = 1;
    const r = t.getBoundingClientRect(); let lx = x + 14, ly = y + 14;
    if (lx + r.width > innerWidth - 8) lx = x - r.width - 14;
    if (ly + r.height > innerHeight - 8) ly = y - r.height - 14;
    t.style.transform = `translate(${lx}px,${ly}px)`;
  }
  function hideTip() { if (tip) tip.style.opacity = 0; }

  function niceStep(range, target) { const raw = range / target, p = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / p; return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p; }
  function niceMax(m) { if (m <= 0) return 1; const st = niceStep(m, 4); return Math.ceil(m / st - 1e-9) * st; }
  function ticks(min, max) { const st = niceStep(max - min, 4), out = []; for (let v = Math.ceil(min / st - 1e-9) * st; v <= max + st * 1e-6; v += st) out.push(+v.toFixed(10)); return out; }
  function xLabelIdx(n, iw, minGap) { const step = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(iw / minGap)))); const idx = []; for (let i = 0; i < n; i += step) idx.push(i); if (n - 1 - idx[idx.length - 1] >= step * .6) idx.push(n - 1); return idx; }

  function frame(host, o) {
    host.innerHTML = '';
    const W = Math.max(280, host.clientWidth || 600), H = o.height || +host.dataset.h || 280;
    const m = Object.assign({ t: 14, r: 14, b: 30, l: 46 }, o.margin || {});
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'dc-svg', role: 'img' }, host);
    if (o.label) svg.setAttribute('aria-label', o.label);
    return { svg, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
  }

  function legend(host, series, pal) {
    if (series.length < 2) return;
    const lg = document.createElement('div'); lg.className = 'dc-legend';
    series.forEach((s, i) => { const it = document.createElement('span'); it.innerHTML = `<i style="background:${s.color || pal[i % pal.length]}${s.dash ? ';background:none;border-top:2px dashed ' + (s.color || pal[i % pal.length]) + ';height:0' : ''}"></i>${s.name}`; lg.appendChild(it); });
    host.appendChild(lg);
  }

  /* ---------- LÍNEAS ---------- */
  function line(host, o) {
    const draw = () => {
      const f = frame(host, o), pal = PAL(), { svg, m, iw, ih } = f;
      const all = o.series.flatMap(s => s.data.filter(v => v != null));
      const lo = o.yMin != null ? o.yMin : Math.min(0, ...all);
      const hi = o.yMax != null ? o.yMax : niceMax(Math.max(...all) * 1.05);
      const n = o.labels.length, X = i => m.l + (n === 1 ? iw / 2 : i * iw / (n - 1)), Y = v => m.t + ih - (v - lo) / (hi - lo) * ih;
      const gg = el('g', { class: 'dc-grid' }, svg);
      ticks(lo, hi).forEach(t => { el('line', { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t) }, gg); const tx = el('text', { x: m.l - 8, y: Y(t) + 4, 'text-anchor': 'end', class: 'dc-ax' }, svg); tx.textContent = fmtNum(t, o.yFmt); });
      xLabelIdx(n, iw, o.labelGap || 70).forEach(i => { const tx = el('text', { x: X(i), y: m.t + ih + 20, 'text-anchor': 'middle', class: 'dc-ax' }, svg); tx.textContent = o.labels[i]; });
      o.series.forEach((s, si) => {
        const c = s.color || pal[si % pal.length]; let d = '', started = false;
        s.data.forEach((v, i) => { if (v == null) { started = false; return; } d += (started ? 'L' : 'M') + X(i).toFixed(1) + ',' + Y(v).toFixed(1); started = true; });
        if (o.area && si === 0) { const pts = s.data.map((v, i) => v == null ? null : [X(i), Y(v)]).filter(Boolean); if (pts.length) el('path', { d: 'M' + pts[0][0] + ',' + Y(lo) + pts.map(p => 'L' + p[0] + ',' + p[1]).join('') + 'L' + pts[pts.length - 1][0] + ',' + Y(lo) + 'Z', fill: c, opacity: .12 }, svg); }
        el('path', { d, fill: 'none', stroke: c, 'stroke-width': s.width || 2, 'stroke-dasharray': s.dash ? '5 4' : '', 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
      });
      const guide = el('line', { y1: m.t, y2: m.t + ih, class: 'dc-guide', opacity: 0 }, svg);
      const dots = o.series.map((s, si) => el('circle', { r: 4, fill: s.color || pal[si % pal.length], opacity: 0 }, svg));
      const hit = el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' }, svg);
      const move = e => {
        const r = svg.getBoundingClientRect(), sx = (e.clientX - r.left) * f.W / r.width;
        const i = Math.max(0, Math.min(n - 1, Math.round((sx - m.l) / (iw / Math.max(1, n - 1)))));
        guide.setAttribute('x1', X(i)); guide.setAttribute('x2', X(i)); guide.setAttribute('opacity', 1);
        let h = `<b>${o.labels[i]}</b>`;
        o.series.forEach((s, si) => { const v = s.data[i]; dots[si].setAttribute('cx', X(i)); dots[si].setAttribute('cy', v == null ? -99 : Y(v)); dots[si].setAttribute('opacity', v == null ? 0 : 1);
          h += `<span><i style="background:${s.color || pal[si % pal.length]}"></i>${s.name}: <b>${v == null ? '—' : (o.tipFmt || o.yFmt ? (o.tipFmt || o.yFmt)(v) : fmtNum(v))}</b></span>`; });
        showTip(h, e.clientX, e.clientY);
      };
      hit.addEventListener('pointermove', move); hit.addEventListener('pointerleave', () => { hideTip(); guide.setAttribute('opacity', 0); dots.forEach(d => d.setAttribute('opacity', 0)); });
      legend(host, o.series, pal);
    };
    draw(); watch(host, draw);
  }

  /* ---------- BARRAS ---------- */
  function bars(host, o) {
    const draw = () => {
      const pal = PAL();
      if (o.horizontal) {
        const rowH = o.rowH || 30, H = o.labels.length * rowH + 10;
        const f = frame(host, Object.assign({}, o, { height: H, margin: Object.assign({ t: 4, r: 56, b: 6, l: o.labelW || 130 }, o.margin || {}) })), { svg, m, iw } = f;
        const tot = o.labels.map((_, i) => o.series.reduce((a, s) => a + (s.data[i] || 0), 0));
        const hi = o.xMax || Math.max(...tot) * 1.02;
        o.labels.forEach((l, i) => {
          const y = m.t + i * rowH; let x = m.l;
          const tx = el('text', { x: m.l - 10, y: y + rowH / 2 + 4, 'text-anchor': 'end', class: 'dc-lab' }, svg); tx.textContent = l;
          el('rect', { x: m.l, y: y + 5, width: iw, height: rowH - 10, rx: 3, class: 'dc-track' }, svg);
          o.series.forEach((s, si) => {
            const v = s.data[i] || 0, w = v / hi * iw;
            const r = el('rect', { x, y: y + 5, width: Math.max(0, w), height: rowH - 10, rx: 3, fill: (s.colors && s.colors[i]) || s.color || pal[si % pal.length] }, svg);
            r.addEventListener('pointermove', e => showTip(`<b>${l}</b><span>${s.name ? s.name + ': ' : ''}<b>${(o.tipFmt || o.xFmt || fmtNum)(v)}</b></span>`, e.clientX, e.clientY));
            r.addEventListener('pointerleave', hideTip); x += w;
          });
          const vt = el('text', { x: x + 8, y: y + rowH / 2 + 4, class: 'dc-val' }, svg); vt.textContent = (o.xFmt || fmtNum)(tot[i]);
        });
        if (o.series.length > 1) legend(host, o.series, pal);
        return;
      }
      const f = frame(host, o), { svg, m, iw, ih } = f;
      const n = o.labels.length, k = o.series.length;
      const tot = o.labels.map((_, i) => o.stacked ? o.series.reduce((a, s) => a + (s.data[i] || 0), 0) : Math.max(...o.series.map(s => s.data[i] || 0)));
      const hi = o.yMax || niceMax(Math.max(...tot) * 1.05), Y = v => m.t + ih - v / hi * ih;
      const gg = el('g', { class: 'dc-grid' }, svg);
      ticks(0, hi).forEach(t => { el('line', { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t) }, gg); const tx = el('text', { x: m.l - 8, y: Y(t) + 4, 'text-anchor': 'end', class: 'dc-ax' }, svg); tx.textContent = fmtNum(t, o.yFmt); });
      const bw = iw / n, gap = Math.min(10, bw * .25), inner = bw - gap;
      const shown = new Set(xLabelIdx(n, iw, o.labelGap || 56));
      o.labels.forEach((l, i) => {
        let base = 0;
        o.series.forEach((s, si) => {
          const v = s.data[i] || 0;
          const w = o.stacked ? inner : inner / k, x = m.l + i * bw + gap / 2 + (o.stacked ? 0 : si * w);
          const y0 = o.stacked ? Y(base + v) : Y(v), h = o.stacked ? Y(base) - Y(base + v) : Y(0) - Y(v);
          const r = el('rect', { x, y: y0, width: Math.max(1, w - (o.stacked ? 0 : 2)), height: Math.max(0, h), rx: 2, fill: (s.colors && s.colors[i]) || s.color || pal[si % pal.length] }, svg);
          r.addEventListener('pointermove', e => { let h2 = `<b>${l}</b>`; o.series.forEach((s2, s2i) => h2 += `<span><i style="background:${s2.color || pal[s2i % pal.length]}"></i>${s2.name ? s2.name + ': ' : ''}<b>${(o.tipFmt || o.yFmt || fmtNum)(s2.data[i] || 0)}</b></span>`); showTip(h2, e.clientX, e.clientY); });
          r.addEventListener('pointerleave', hideTip);
          base += v;
        });
        if (shown.has(i)) { const tx = el('text', { x: m.l + i * bw + bw / 2, y: m.t + ih + 20, 'text-anchor': 'middle', class: 'dc-ax' }, svg); tx.textContent = l; }
      });
      legend(host, o.series, pal);
    };
    draw(); watch(host, draw);
  }

  /* ---------- MAPA DE CALOR ---------- */
  function heat(host, o) {
    const draw = () => {
      const rows = o.rows.length, cols = o.cols.length;
      const lw = o.labelW || 70, cell = o.cellH || 26;
      const H = rows * cell + (o.colLabels === false ? 8 : 30);
      const f = frame(host, Object.assign({}, o, { height: H, margin: { t: 4, r: 6, b: o.colLabels === false ? 4 : 26, l: lw } })), { svg, m, iw } = f;
      const cw = iw / cols, vals = o.values.flat().filter(v => v != null);
      const lo = o.min != null ? o.min : Math.min(...vals), hi = o.max != null ? o.max : Math.max(...vals);
      const base = o.color || css('--c1') || '#c5f23a';
      o.rows.forEach((rl, r) => {
        const tx = el('text', { x: m.l - 8, y: m.t + r * cell + cell / 2 + 4, 'text-anchor': 'end', class: 'dc-lab' }, svg); tx.textContent = rl;
        o.cols.forEach((cl, c) => {
          const v = o.values[r][c]; if (v == null) return;
          const t = hi === lo ? 1 : (v - lo) / (hi - lo);
          const rect = el('rect', { x: m.l + c * cw + 1, y: m.t + r * cell + 1, width: Math.max(1, cw - 2), height: cell - 2, rx: 2, fill: base, 'fill-opacity': (.06 + .94 * t).toFixed(3) }, svg);
          if (o.showValues && cw > 30) { const vt = el('text', { x: m.l + c * cw + cw / 2, y: m.t + r * cell + cell / 2 + 4, 'text-anchor': 'middle', class: 'dc-cell' + (t > .55 ? ' dark' : '') }, svg); vt.textContent = (o.fmt || fmtNum)(v); }
          rect.addEventListener('pointermove', e => showTip(`<b>${rl} · ${cl}</b><span>${o.valueName || ''} <b>${(o.fmt || fmtNum)(v)}</b></span>`, e.clientX, e.clientY));
          rect.addEventListener('pointerleave', hideTip);
        });
      });
      if (o.colLabels !== false) {
        const step = Math.max(1, Math.ceil(cols / Math.max(2, Math.floor(iw / 34))));
        o.cols.forEach((cl, c) => { if (c % step) return; const tx = el('text', { x: m.l + c * cw + cw / 2, y: m.t + rows * cell + 18, 'text-anchor': 'middle', class: 'dc-ax' }, svg); tx.textContent = cl; });
      }
    };
    draw(); watch(host, draw);
  }

  /* ---------- re-render al cambiar el tamaño ---------- */
  const registry = new Map();
  function watch(host, draw) { registry.set(host, draw); host._draw = draw; }
  let lastW = innerWidth, rt;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if (Math.abs(innerWidth - lastW) < 2) return; lastW = innerWidth; registry.forEach(d => d()); }, 150); });

  g.DC = { line, bars, heat, redraw: () => registry.forEach(d => d()) };
})(window);
