/* dash.js — idioma ES/EN compartido por los dashboards */
(function () {
  const store = { get: k => { try { return localStorage.getItem(k) } catch (e) { return null } }, set: (k, v) => { try { localStorage.setItem(k, v) } catch (e) { } } };
  const tx = [...document.querySelectorAll('[data-en]')]; tx.forEach(e => e._es = e.innerHTML);
  window.LANG = 'es';
  window.L = (es, en) => window.LANG === 'en' ? en : es;
  const fire = () => document.dispatchEvent(new CustomEvent('langchange', { detail: window.LANG }));
  function apply(l) {
    window.LANG = l; document.documentElement.lang = l;
    tx.forEach(e => e.innerHTML = l === 'en' ? e.dataset.en : e._es);
    document.querySelectorAll('.lang button').forEach(b => b.setAttribute('aria-pressed', b.dataset.lang === l));
    const t = document.querySelector('meta[name="title-en"]');
    if (t) { if (!document._esTitle) document._esTitle = document.title; document.title = l === 'en' ? t.content : document._esTitle; }
    fire();
  }
  document.querySelectorAll('.lang button').forEach(b => b.addEventListener('click', () => { store.set('mt-lang', b.dataset.lang); apply(b.dataset.lang); }));
  const q = new URLSearchParams(location.search).get('lang');
  window.addEventListener('DOMContentLoaded', () => apply(q === 'en' || q === 'es' ? q : (store.get('mt-lang') || ((navigator.language || 'es').toLowerCase().startsWith('es') ? 'es' : 'en'))));
  window.fmt = {
    int: v => Math.round(v).toLocaleString(window.LANG === 'en' ? 'en-US' : 'es-AR'),
    usd: v => 'US$ ' + Math.round(v).toLocaleString(window.LANG === 'en' ? 'en-US' : 'es-AR'),
    usdk: v => 'US$ ' + (Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : (v / 1e3).toFixed(0) + 'k'),
    pct: (v, d = 1) => (v * 100).toFixed(d).replace('.', window.LANG === 'en' ? '.' : ',') + '%',
    dec: (v, d = 2) => v.toFixed(d).replace('.', window.LANG === 'en' ? '.' : ','),
  };
})();
