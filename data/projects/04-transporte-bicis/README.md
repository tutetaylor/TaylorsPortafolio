# Pronóstico de demanda de bicicletas públicas en Buenos Aires
**Data Scientist · Transporte** · Mateo Taylor

🔗 [Dashboard interactivo](https://www.taylorsportafolio.com/data/bicis/) · 📓 [Notebook](analisis.ipynb)

## Pregunta
Para que haya bicis donde y cuando se necesitan, la operación se planifica con un día de anticipación. ¿Cuántos viajes habrá mañana, hora por hora?

## Qué hice
- Exploración de patrones por hora y día, y efecto de la lluvia, la temperatura, los feriados y las vacaciones.
- Variables sin fuga de información: solo datos disponibles el día anterior.
- **Validación temporal** (prueba con 4 meses no vistos).
- Comparación con dos baselines, **Ridge con splines** y **gradient boosting con pérdida de Poisson**.
- Análisis del error por hora y por intensidad de lluvia; importancia de variables.
- **Escenarios:** cuánto cae la demanda si llueve a la tarde.

## Resultados
- 25.000 viajes por día, un 12 % más en 2025; días hábiles con picos a las 8 y 18 h.
- Con lluvia fuerte la demanda cae al 34 % de lo normal.
- Gradient boosting reduce el error un 39 % frente a "igual que la semana pasada" (WAPE 16 % vs 26 %). Un modelo lineal no alcanza.
- Con lluvia fuerte, el error del modelo es 19 % contra 169 % del baseline.
- Una tarde de lluvia moderada baja los viajes del día un 27 %.

## Archivos
| Archivo | Qué es |
|---|---|
| `analisis.ipynb` | Notebook ejecutado con el análisis completo, gráficos y conclusiones |
| `analisis.py` | El mismo análisis como script (celdas `# %%`) |
| `generar_datos.py` | Genera el dataset simulado en `data/` |
| `data/` | viajes_por_hora.csv.gz |

**Herramientas:** Python · pandas · scikit-learn · matplotlib

## Sobre los datos
Datos **simulados** con reglas de negocio realistas; no representan a ninguna empresa real. Para pasar a datos reales alcanza con reemplazar los archivos de `data/` por otros con el mismo esquema, por ejemplo los datos abiertos de viajes de EcoBici (Buenos Aires Data) agregados por hora, más un histórico meteorológico.

## Cómo ejecutarlo
```bash
pip install pandas numpy scikit-learn scipy matplotlib
python generar_datos.py
jupyter notebook analisis.ipynb
```
