# Embudo de conversión y A/B test del checkout
**Data Analyst · Marketing** · Mateo Taylor

🔗 [Dashboard interactivo](https://www.taylorsportafolio.com/data/marketing/) · 📓 [Notebook](analisis.ipynb)

## Pregunta
Antes de pedir más tráfico: ¿dónde se pierde la gente en el embudo y el nuevo checkout de una sola página mejora la conversión de verdad o es ruido?

## Qué hice
- Detección y limpieza de **tráfico de bots**.
- Embudo completo y por canal (UTM) y dispositivo; ingreso por sesión.
- **Diseño del experimento:** hipótesis, métrica principal y de control, cálculo de **potencia** previo.
- Control de **SRM** (sample ratio mismatch).
- **Test de dos proporciones** con intervalo de confianza y **bootstrap** del ingreso por checkout.
- Efecto por dispositivo con corrección de Bonferroni y estabilidad en el tiempo.
- Impacto proyectado con su rango de incertidumbre.

## Resultados
- El 2,4 % del tráfico eran bots de un scraper de precios.
- La mayor fuga está en la página de producto: el 81 % no agrega al carrito.
- Paid social trae el 33 % de las sesiones y solo el 12 % de los ingresos.
- El checkout de una página sube la conversión de 53,6 % a 58,7 % (+5,1 pp, IC 95 % +2,6 a +7,6; p < 0,001), sin bajar el ticket.
- El efecto es casi todo de mobile (+9,0 pp). Impacto esperado: ~225 compras y ~US$ 14.000 más por mes.

## Archivos
| Archivo | Qué es |
|---|---|
| `analisis.ipynb` | Notebook ejecutado con el análisis completo, gráficos y conclusiones |
| `analisis.py` | El mismo análisis como script (celdas `# %%`) |
| `generar_datos.py` | Genera el dataset simulado en `data/` |
| `data/` | sesiones.csv.gz |

**Herramientas:** Python · pandas · SciPy · matplotlib

## Sobre los datos
Datos **simulados** con reglas de negocio realistas; no representan a ninguna empresa real. Para pasar a datos reales alcanza con reemplazar los archivos de `data/` por otros con el mismo esquema, por ejemplo un export de eventos de GA4 (BigQuery) o la base de pedidos con la asignación del experimento.

## Cómo ejecutarlo
```bash
pip install pandas numpy scikit-learn scipy matplotlib
python generar_datos.py
jupyter notebook analisis.ipynb
```
