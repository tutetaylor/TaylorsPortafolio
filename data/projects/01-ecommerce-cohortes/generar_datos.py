"""
Genera el dataset SIMULADO de una tienda de indumentaria online (Argentina, 2024-2025).

No son datos reales de ninguna empresa: están construidos con reglas de negocio
plausibles (estacionalidad, canales con distinta retención, cupones, devoluciones)
y con algunos errores de carga a propósito, para practicar la limpieza.

Salida (carpeta data/):
  clientes.csv       un registro por cliente
  pedidos.csv        un registro por pedido
  items.csv          un registro por producto dentro de cada pedido
  marketing.csv      inversión mensual por canal pago (USD)

Uso:  python generar_datos.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

MESES = pd.period_range("2024-01", "2025-12", freq="M")
ESTACIONAL = {1: .85, 2: .80, 3: .95, 4: 1.0, 5: 1.35, 6: 1.05, 7: 1.12, 8: .95, 9: 1.0, 10: 1.08, 11: 1.60, 12: 1.45}

CANALES = ["Orgánico", "Paid Social", "Google Ads", "Email", "Referidos"]
# share de adquisición al inicio y al final del período (Paid Social crece)
MIX_INI = np.array([.34, .22, .24, .08, .12])
MIX_FIN = np.array([.24, .38, .22, .06, .10])
# churn mensual y prob. de compra mensual de clientes activos, por canal
CHURN = {"Orgánico": .15, "Paid Social": .27, "Google Ads": .19, "Email": .10, "Referidos": .11}
COMPRA = {"Orgánico": .20, "Paid Social": .14, "Google Ads": .17, "Email": .24, "Referidos": .23}
CAC_OBJ = {"Paid Social": 30.0, "Google Ads": 26.0, "Email": 9.0, "Referidos": 14.0}

REGIONES = ["CABA", "GBA", "Córdoba", "Santa Fe", "Mendoza", "Resto del país"]
P_REGION = [.30, .29, .12, .10, .06, .13]

CATEG = {  # nombre: (precio medio USD, peso en el mix, tasa de devolución)
    "Remeras": (19, .30, .04), "Buzos": (39, .17, .05), "Pantalones": (44, .17, .07),
    "Zapatillas": (86, .13, .11), "Camperas": (96, .08, .06), "Accesorios": (14, .15, .02),
}
CAT_N = list(CATEG)
CAT_P = np.array([v[1] for v in CATEG.values()]); CAT_P = CAT_P / CAT_P.sum()

clientes, pedidos, items = [], [], []
cid = 0; oid = 100000

for mi, mes in enumerate(MESES):
    t = mi / (len(MESES) - 1)
    mix = MIX_INI * (1 - t) + MIX_FIN * t
    nuevos = int(rng.poisson(175 * (1.028 ** mi) * ESTACIONAL[mes.month]))
    canales = rng.choice(CANALES, size=nuevos, p=mix / mix.sum())
    for canal in canales:
        cid += 1
        cliente = f"C{cid:05d}"
        dia = rng.integers(1, mes.days_in_month + 1)
        alta = pd.Timestamp(mes.year, mes.month, dia)
        region = rng.choice(REGIONES, p=P_REGION)
        cupon = rng.random() < (.62 if canal == "Paid Social" else .30 if canal == "Google Ads" else .15)
        clientes.append((cliente, alta.date(), canal, region, rng.choice(["F", "M", "X"], p=[.55, .43, .02]),
                         int(rng.integers(18, 56))))
        churn = CHURN[canal] + (.07 if cupon else 0)
        compra = COMPRA[canal]
        # primer pedido (el día de alta) + recompras mes a mes mientras siga activo
        fechas = [(alta, cupon)]
        for mj in range(mi + 1, len(MESES)):
            if rng.random() < churn:
                break
            m2 = MESES[mj]
            if rng.random() < compra * ESTACIONAL[m2.month] ** .8:
                fechas.append((pd.Timestamp(m2.year, m2.month, int(rng.integers(1, m2.days_in_month + 1))), rng.random() < .12))
        for k, (f, usa_cupon) in enumerate(fechas):
            oid += 1
            evento = (f.month == 5 and 10 <= f.day <= 16) or (f.month == 11 and 1 <= f.day <= 8)
            n_items = 1 + rng.poisson(.75 if k == 0 else 1.0)
            dto = .25 if evento else (.15 if usa_cupon else 0.0)
            estado = "cancelado" if rng.random() < .03 else "completado"
            total = 0.0
            for _ in range(n_items):
                cat = rng.choice(CAT_N, p=CAT_P)
                precio = round(float(CATEG[cat][0] * rng.lognormal(0, .22)), 2)
                cant = 1 if rng.random() < .9 else 2
                devuelto = rng.random() < CATEG[cat][2]
                items.append((oid, cat, cant, precio, devuelto))
                total += precio * cant
            envio = 0.0 if total >= 80 else 6.5
            pedidos.append((oid, cliente, f.date(), estado, round(total * (1 - dto), 2), round(total * dto, 2), envio,
                            "cupón" if usa_cupon else ("evento" if evento else "ninguno")))

cli = pd.DataFrame(clientes, columns=["cliente_id", "fecha_alta", "canal_adquisicion", "region", "genero", "edad"])
ped = pd.DataFrame(pedidos, columns=["pedido_id", "cliente_id", "fecha", "estado", "total_usd", "descuento_usd", "envio_usd", "tipo_descuento"])
it = pd.DataFrame(items, columns=["pedido_id", "categoria", "cantidad", "precio_unit_usd", "devuelto"])

# inversión mensual en marketing (canales pagos), según altas reales x CAC objetivo con ruido
alta_mes = pd.to_datetime(cli["fecha_alta"]).dt.to_period("M")
mk = []
for mes in MESES:
    for canal, cac in CAC_OBJ.items():
        n = int(((alta_mes == mes) & (cli["canal_adquisicion"] == canal)).sum())
        ruido = rng.normal(1, .08) * (1.12 if mes.month in (11, 12) else 1)  # pauta más cara en fin de año
        mk.append((str(mes), canal, round(n * cac * ruido, 2)))
mkt = pd.DataFrame(mk, columns=["mes", "canal", "inversion_usd"])

# ---- errores de carga a propósito (para la etapa de limpieza) ----
dup = ped.sample(frac=.006, random_state=1)                       # pedidos duplicados
ped = pd.concat([ped, dup]).sort_values("pedido_id").reset_index(drop=True)
idx = cli.sample(frac=.012, random_state=2).index; cli.loc[idx, "region"] = np.nan   # región vacía
alias = {"CABA": ["caba", "Capital Federal", "C.A.B.A."], "Córdoba": ["Cordoba", "cordoba"]}
for real, variantes in alias.items():
    idx = cli[cli["region"] == real].sample(frac=.05, random_state=3).index
    cli.loc[idx, "region"] = rng.choice(variantes, size=len(idx))
idx = ped.sample(n=25, random_state=4).index; ped.loc[idx, "total_usd"] = 0.0         # totales en cero (pruebas)

cli.to_csv(OUT / "clientes.csv", index=False)
ped.to_csv(OUT / "pedidos.csv", index=False)
it.to_csv(OUT / "items.csv", index=False)
mkt.to_csv(OUT / "marketing.csv", index=False)
print(f"clientes {len(cli):,} · pedidos {len(ped):,} · ítems {len(it):,} · marketing {len(mkt):,}")
