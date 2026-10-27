"""
Arma los dashboards de data/<proyecto>/index.html a partir de src.html:
  - [[texto en español¦english text]]  ->  <x-t data-en="...">texto</x-t>  (cambio de idioma)
  - /*DATA*/                           ->  window.DATA = {contenido de data.json}
Uso: python tools/build_dashboards.py
"""
import html, json, re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent  # carpeta data/ (se publica en /data/)


def i18n(src):
    src = re.sub(r'(alt|aria-label|title)="\[\[(.*?)¦(.*?)\]\]"',
                 lambda m: f'{m.group(1)}="{m.group(2)}"', src)
    return re.sub(r"\[\[(.*?)¦(.*?)\]\]",
                  lambda m: f'<x-t data-en="{html.escape(m.group(2), quote=True)}">{m.group(1)}</x-t>', src, flags=re.S)


for src_path in sorted(DOCS.glob("*/src.html")) + ([DOCS / "src.html"] if (DOCS / "src.html").exists() else []):
    src = src_path.read_text(encoding="utf-8")
    out = i18n(src)
    assert "¦" not in out, f"marcador sin cerrar en {src_path}"
    data_path = src_path.parent / "data.json"
    if "/*DATA*/" in out:
        data = data_path.read_text(encoding="utf-8")
        out = out.replace("/*DATA*/", "window.DATA = " + data + ";")
    (src_path.parent / "index.html").write_text(out, encoding="utf-8")
    print("OK", src_path.parent.name or "docs", f"{len(out)/1024:.0f} KB")
