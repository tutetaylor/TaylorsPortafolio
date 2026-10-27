"""
Genera el dataset SIMULADO de sesiones web de una tienda online (8 semanas, sep–oct 2025).

Incluye el embudo completo (visita → producto → carrito → checkout → compra), canales con
UTM, dispositivo y un experimento A/B en el checkout:
  A (control):  checkout en 3 pasos
  B (variante): checkout en una sola página con pago express
El experimento corre desde el 15/09; a partir de esa fecha, cada sesión que inicia el checkout
se asigna al azar a A o B (50/50).

También agrega tráfico de bots (sesiones de 0 segundos con muchas páginas) para limpiar.
No son datos reales de ninguna empresa.

Uso:  python generar_datos.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)
OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)

DIAS = pd.date_range("2025-09-01", "2025-10-26", freq="D")
INICIO_TEST = pd.Timestamp("2025-09-15")

CANALES = {  # (source, medium): (share de sesiones, multiplicador de intención)
    ("google", "cpc"): (.22, 1.10), ("instagram", "paid_social"): (.24, .70), ("tiktok", "paid_social"): (.09, .55),
    ("google", "organic"): (.18, 1.15), ("newsletter", "email"): (.08, 1.65), ("(direct)", "(none)"): (.14, 1.30),
    ("blog", "referral"): (.05, .95),
}
FUENTES = list(CANALES); P_CANAL = np.array([v[0] for v in CANALES.values()]); P_CANAL /= P_CANAL.sum()
DISP = {"mobile": (.68, .85), "desktop": (.28, 1.30), "tablet": (.04, 1.0)}
D_N = list(DISP); D_P = [v[0] for v in DISP.values()]

# probabilidades base de avanzar de cada etapa a la siguiente
P_PROD, P_CARR, P_CHECK = .58, .17, .52
P_COMPRA = {"A": {"mobile": .43, "desktop": .58, "tablet": .50},
            "B": {"mobile": .50, "desktop": .59, "tablet": .53}}   # la variante ayuda sobre todo en mobile

rows = []
sid = 0
for dia in DIAS:
    finde = dia.dayofweek >= 5
    n = int(rng.poisson(2050 * (1.18 if finde else 1.0) * (1 + .004 * (dia - DIAS[0]).days)))
    canal_idx = rng.choice(len(FUENTES), size=n, p=P_CANAL)
    disp = rng.choice(D_N, size=n, p=D_P)
    nuevo = rng.random(n) < .63
    for i in range(n):
        sid += 1
        src, med = FUENTES[canal_idx[i]]
        intent = CANALES[(src, med)][1] * DISP[disp[i]][1] * (.85 if nuevo[i] else 1.25)
        dur = int(rng.gamma(2.0, 70 * intent))
        paginas = 1 + rng.poisson(2.2 * intent)
        vio = rng.random() < min(.95, P_PROD * intent ** .5)
        carr = vio and rng.random() < min(.9, P_CARR * intent)
        chk = carr and rng.random() < min(.95, P_CHECK * intent ** .3)
        grupo = None
        if chk and dia >= INICIO_TEST:
            grupo = "A" if rng.random() < .5 else "B"
        p_c = P_COMPRA[grupo or "A"][disp[i]] * (intent ** .25)
        compra = chk and rng.random() < min(.97, p_c)
        ingreso = round(float(rng.lognormal(np.log(78), .45)), 2) if compra else 0.0
        rows.append((f"S{sid:07d}", dia.date(), src, med, disp[i], bool(nuevo[i]), dur, paginas,
                     vio, carr, chk, compra, ingreso, grupo))

df = pd.DataFrame(rows, columns=["sesion_id", "fecha", "utm_source", "utm_medium", "dispositivo", "usuario_nuevo",
                                 "duracion_seg", "paginas_vistas", "vio_producto", "agrego_carrito",
                                 "inicio_checkout", "compro", "ingreso_usd", "grupo_test"])

# tráfico de bots: una IP de scraping que entra como "referral"
n_bots = int(len(df) * .025)
bots = pd.DataFrame({
    "sesion_id": [f"B{i:07d}" for i in range(n_bots)], "fecha": rng.choice(DIAS.date, n_bots),
    "utm_source": "price-crawler.net", "utm_medium": "referral", "dispositivo": "desktop", "usuario_nuevo": True,
    "duracion_seg": 0, "paginas_vistas": rng.integers(25, 80, n_bots), "vio_producto": True, "agrego_carrito": False,
    "inicio_checkout": False, "compro": False, "ingreso_usd": 0.0, "grupo_test": None})
df = pd.concat([df, bots]).sample(frac=1, random_state=3).sort_values("fecha", kind="stable").reset_index(drop=True)
df.to_csv(OUT / "sesiones.csv.gz", index=False, compression="gzip")
print(f"sesiones: {len(df):,} (bots: {n_bots:,}) · compras: {df['compro'].sum():,} · en el test: {df['grupo_test'].notna().sum():,}")
