"""
Mini ejecutor de notebooks (sin Jupyter).

Toma un script con celdas marcadas al estilo "percent" de VS Code / Jupytext:

    # %% [markdown]
    # # Título
    # Texto en markdown...

    # %%
    import pandas as pd
    df = pd.read_csv("data/ventas.csv")
    df.head()

Ejecuta las celdas en orden, en un mismo namespace, y guarda un .ipynb
(nbformat 4) con las salidas: texto impreso, el valor de la última expresión
(las tablas de pandas se guardan como HTML) y los gráficos de matplotlib como PNG.

Uso:  python tools/run_notebook.py proyecto/analisis.py proyecto/analisis.ipynb
"""
import ast, base64, contextlib, io, json, os, sys, traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_cells(src):
    cells, cur, kind = [], [], None
    for line in src.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, cur))
            kind = "markdown" if "[markdown]" in line else "code"
            cur = []
        elif kind is not None:
            cur.append(line)
    if kind is not None:
        cells.append((kind, cur))
    out = []
    for kind, lines in cells:
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l) for l in lines]
        out.append((kind, "\n".join(lines)))
    return out


def as_source(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def rich_repr(value):
    data = {"text/plain": as_source(repr(value))}
    html = None
    try:
        import pandas as pd
        if isinstance(value, (pd.DataFrame, pd.Series)):
            obj = value.to_frame() if isinstance(value, pd.Series) else value
            html = obj.to_html(max_rows=30, border=0, classes="dataframe")
            data["text/plain"] = as_source(value.to_string(max_rows=30))
        elif hasattr(value, "to_html") and type(value).__name__ == "Styler":
            html = value.to_html()
    except Exception:
        pass
    if html:
        data["text/html"] = as_source(html)
    return data


def run(src_path, out_path):
    src = open(src_path, encoding="utf-8").read()
    workdir = os.path.dirname(os.path.abspath(src_path))
    os.chdir(workdir)
    sys.path.insert(0, workdir)
    ns = {"__name__": "__main__"}
    nb_cells, count, failed = [], 0, False

    for kind, text in parse_cells(src):
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": as_source(text)})
            continue
        count += 1
        outputs = []
        buf = io.StringIO()
        try:
            tree = ast.parse(text)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<celda {count}>", "exec"), ns)
                value = eval(compile(last, f"<celda {count}>", "eval"), ns) if last is not None else None
            if buf.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": as_source(buf.getvalue())})
            for num in plt.get_fignums():
                fig = plt.figure(num)
                png = io.BytesIO()
                fig.savefig(png, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": base64.b64encode(png.getvalue()).decode(),
                                         "text/plain": ["<Figure>"]}})
            plt.close("all")
            if value is not None and not (hasattr(value, "__module__") and "matplotlib" in str(type(value))):
                outputs.append({"output_type": "execute_result", "execution_count": count,
                                "metadata": {}, "data": rich_repr(value)})
        except Exception as e:
            failed = True
            if buf.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": as_source(buf.getvalue())})
            tb = traceback.format_exc().splitlines()
            outputs.append({"output_type": "error", "ename": type(e).__name__, "evalue": str(e), "traceback": tb})
            print(f"ERROR en celda {count}:\n" + "\n".join(tb), file=sys.stderr)
        nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {},
                         "outputs": outputs, "source": as_source(text)})
        if failed:
            break

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}},
          "cells": nb_cells}
    with open(out_path if os.path.isabs(out_path) else os.path.join(os.getcwd(), os.path.basename(out_path)), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return not failed


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    out = os.path.abspath(out)
    ok = run(src, out)
    print(("OK  " if ok else "FALLÓ ") + out)
    sys.exit(0 if ok else 1)
