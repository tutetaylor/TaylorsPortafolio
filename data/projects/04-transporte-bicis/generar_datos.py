"""
Genera el dataset SIMULADO de viajes por hora de un sistema de bicicletas públicas en Buenos Aires
(2024–2025), inspirado en el funcionamiento de EcoBici, junto con el clima de cada hora.

La demanda depende de la hora y el día (picos de ida y vuelta al trabajo), de los feriados argentinos,
de las vacaciones de verano, de la temperatura, de la lluvia y de una ampliación de estaciones a mitad
de 2025. No son datos oficiales: el script está pensado para reemplazarse por los datos abiertos reales
de viajes y por un histórico meteorológico.

Uso:  python generar_datos.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(11)
OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)

idx = pd.date_range("2024-01-01 00:00", "2025-12-31 23:00", freq="h")
df = pd.DataFrame({"fecha_hora": idx})
h, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values

FERIADOS = pd.to_datetime([
    "2024-01-01", "2024-02-12", "2024-02-13", "2024-03-24", "2024-03-28", "2024-03-29", "2024-04-02", "2024-05-01",
    "2024-05-25", "2024-06-17", "2024-06-20", "2024-06-21", "2024-07-09", "2024-08-17", "2024-10-11", "2024-11-18",
    "2024-12-08", "2024-12-25", "2025-01-01", "2025-03-03", "2025-03-04", "2025-03-24", "2025-04-02", "2025-04-17",
    "2025-04-18", "2025-05-01", "2025-05-02", "2025-05-25", "2025-06-16", "2025-06-20", "2025-07-09", "2025-08-15",
    "2025-10-12", "2025-11-21", "2025-11-24", "2025-12-08", "2025-12-25"]).date
feriado = np.isin(idx.date, FERIADOS)
finde = (dow >= 5) | feriado

# ---- clima (hemisferio sur: calor en enero, frío en julio) ----
estacional = 18 + 7.5 * np.cos(2 * np.pi * (doy - 20) / 365.25)
diario = 4.5 * np.cos(2 * np.pi * (h - 15) / 24)
anomalia = np.convolve(rng.normal(0, 1, len(idx) + 200), np.ones(96) / 96 * 9.5, "same")[:len(idx)]
temp = estacional + diario + anomalia + rng.normal(0, .6, len(idx))
# lluvia: episodios de 2 a 14 horas, más frecuentes en verano y otoño
lluvia = np.zeros(len(idx))
p_inicio = .010 + .006 * np.cos(2 * np.pi * (doy - 40) / 365.25)
i = 0
while i < len(idx):
    if rng.random() < p_inicio[i]:
        dur = rng.integers(2, 15); intens = rng.gamma(1.6, 2.2)
        lluvia[i:i + dur] = np.clip(intens * rng.lognormal(0, .6, min(dur, len(idx) - i)), 0, 45)
        i += dur
    else:
        i += 1
humedad = np.clip(68 - 1.1 * (temp - 18) + 18 * (lluvia > 0) + rng.normal(0, 7, len(idx)), 25, 100)
viento = np.clip(rng.gamma(2.4, 5.2, len(idx)) + 6 * (lluvia > 0), 0, 70)

# ---- demanda ----
laboral = np.exp(-((h - 8.2) ** 2) / 2.2) * 1.0 + np.exp(-((h - 18.3) ** 2) / 3.0) * 1.15 + np.exp(-((h - 13) ** 2) / 6) * .45
laboral += .06 + .35 * ((h >= 7) & (h <= 21))
descanso = np.exp(-((h - 15) ** 2) / 14) * 1.05 + .05 + .12 * ((h >= 9) & (h <= 21))
perfil = np.where(finde, descanso * .78, laboral)
viernes = np.where(dow == 4, 1 + .12 * (h >= 19), 1)
tend = 1 + .11 * (idx - idx[0]).days.values / 365
ampliacion = np.where(idx >= "2025-07-01", 1.08, 1.0)
vacaciones = np.where((idx.month == 1) & ~finde, .82, 1.0) * np.where((idx.month == 7) & (idx.day >= 15) & (idx.day <= 28) & ~finde, .9, 1)
f_temp = np.exp(-((temp - 23) ** 2) / (2 * 9.5 ** 2))
f_lluvia = np.exp(-.17 * np.minimum(lluvia, 15) ** .8) * np.where(lluvia > 0, .82, 1)
f_viento = np.exp(-.006 * np.maximum(viento - 25, 0))
mu = 2150 * perfil * viernes * tend * ampliacion * vacaciones * (.35 + .65 * f_temp) * f_lluvia * f_viento
viajes = rng.negative_binomial(40, 40 / (40 + mu))

df["viajes"] = viajes
df["temperatura_c"] = temp.round(1); df["lluvia_mm"] = lluvia.round(1)
df["humedad_pct"] = humedad.round(0).astype(int); df["viento_kmh"] = viento.round(1)
df["feriado"] = feriado
df.to_csv(OUT / "viajes_por_hora.csv.gz", index=False, compression="gzip")
print(f"horas: {len(df):,} · viajes totales: {df['viajes'].sum():,} · promedio diario: {df['viajes'].sum() / (len(df) / 24):,.0f}")
