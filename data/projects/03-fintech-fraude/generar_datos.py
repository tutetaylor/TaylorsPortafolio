"""
Genera el dataset SIMULADO de transacciones con tarjeta de una fintech (ene–jun 2025).

Cada transacción tiene datos del comercio, del canal, del historial reciente del cliente y la
etiqueta `fraude` (1 = contracargo confirmado). El fraude se genera con un modelo de riesgo
latente con interacciones y ruido, de forma que ninguna variable sola lo explique.
No son datos reales de ninguna entidad.

Uso:  python generar_datos.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(2025)
OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)
N, N_CLIENTES = 90_000, 7_000

COMERCIOS = {  # categoría: (peso, monto medio USD, riesgo base)
    "Supermercado": (.24, 38, -1.2), "Gastronomía": (.17, 24, -.9), "Combustible": (.10, 42, -1.0),
    "Indumentaria": (.10, 55, -.2), "Electrónica": (.07, 210, 1.0), "E-commerce": (.13, 60, .6),
    "Viajes": (.04, 380, .9), "Suscripciones digitales": (.08, 12, .1), "Billeteras / recargas": (.07, 30, .8),
}
CAT = list(COMERCIOS); P_CAT = np.array([v[0] for v in COMERCIOS.values()]); P_CAT /= P_CAT.sum()

clientes = pd.DataFrame({
    "cliente_id": [f"T{i:05d}" for i in range(N_CLIENTES)],
    "gasto_medio": rng.lognormal(np.log(45), .5, N_CLIENTES),
    "antiguedad_dias": rng.integers(10, 2500, N_CLIENTES),
    "edad": rng.integers(18, 75, N_CLIENTES),
})

idx = rng.integers(0, N_CLIENTES, N)
c = clientes.iloc[idx].reset_index(drop=True)
fecha = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 181 * 24 * 60, N), unit="m")
cat = rng.choice(CAT, N, p=P_CAT)
canal = np.where(np.isin(cat, ["E-commerce", "Suscripciones digitales", "Viajes", "Billeteras / recargas"]),
                 np.where(rng.random(N) < .9, "online", "presencial"),
                 np.where(rng.random(N) < .55, "contactless", np.where(rng.random(N) < .85, "presencial", "online")))
monto = np.array([COMERCIOS[k][1] for k in cat]) * rng.lognormal(0, .7, N) * (c["gasto_medio"].values / 45) ** .5
internacional = rng.random(N) < np.where(canal == "online", .14, .03)
distancia = np.where(canal == "online", 0, rng.gamma(1.3, 6, N))
disp_nuevo = (canal == "online") & (rng.random(N) < .12)
tx_24h = rng.poisson(2.1, N)
fallidos = rng.poisson(.08, N)
hora = fecha.hour.values

# ---- riesgo latente de fraude ----
nocturno = (hora >= 0) & (hora < 6)
ratio = monto / c["gasto_medio"].values
z = (-7.25 + np.array([COMERCIOS[k][2] for k in cat])
     + 1.15 * np.log1p(ratio) + .9 * internacional + 1.1 * disp_nuevo + .75 * nocturno
     + .32 * np.clip(tx_24h - 3, 0, None) + .9 * fallidos + 1.0 * (c["antiguedad_dias"].values < 60)
     + .6 * (distancia > 40) + .8 * (internacional & disp_nuevo) + .5 * (nocturno & (canal == "online"))
     + rng.normal(0, .9, N))
fraude = rng.random(N) < 1 / (1 + np.exp(-z))

# el fraude real deja huellas: más transacciones en poco tiempo y montos de prueba
tx_24h = np.where(fraude & (rng.random(N) < .45), tx_24h + rng.integers(3, 9, N), tx_24h)
prueba = fraude & (rng.random(N) < .18)
monto = np.where(prueba, rng.uniform(.5, 3, N), monto)

df = pd.DataFrame({
    "transaccion_id": [f"X{i:07d}" for i in range(N)], "fecha_hora": fecha, "cliente_id": c["cliente_id"].values,
    "monto_usd": monto.round(2), "categoria_comercio": cat, "canal": canal, "internacional": internacional,
    "distancia_domicilio_km": distancia.round(1), "dispositivo_nuevo": disp_nuevo, "tx_ultimas_24h": tx_24h,
    "intentos_fallidos_previos": fallidos, "gasto_medio_30d_usd": c["gasto_medio"].values.round(2),
    "antiguedad_cuenta_dias": c["antiguedad_dias"].values, "edad_titular": c["edad"].values, "fraude": fraude.astype(int),
}).sort_values("fecha_hora").reset_index(drop=True)
df.to_csv(OUT / "transacciones.csv.gz", index=False, compression="gzip")
print(f"transacciones: {len(df):,} · fraudes: {df['fraude'].sum():,} ({df['fraude'].mean():.2%}) · "
      f"monto en fraude: US$ {df.loc[df['fraude']==1, 'monto_usd'].sum():,.0f}")
