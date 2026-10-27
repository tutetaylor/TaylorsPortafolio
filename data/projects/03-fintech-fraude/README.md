# Detección de fraude en pagos con tarjeta
**Data Scientist · Fintech** · Mateo Taylor

🔗 [Dashboard interactivo](https://www.taylorsportafolio.com/data/fraude/) · 📓 [Notebook](analisis.ipynb)

## Pregunta
Una fintech pierde con cada fraude que no detecta y también con cada cliente legítimo que bloquea. ¿Cómo priorizar mejor las alertas que la regla fija actual, eligiendo el umbral según el costo real de cada error?

## Qué hice
- Exploración de patrones de fraude por hora, rubro, monto y comportamiento.
- Ingeniería de variables de comportamiento anómalo (monto vs. gasto habitual, ráfagas, micro-cargos).
- **Partición temporal** (entrenamiento ene–abr, validación may, prueba jun).
- Comparación: regla actual, **regresión logística**, **random forest** y **gradient boosting**, con **PR-AUC**.
- Umbral óptimo según costo y **decisión por pérdida esperada**, con verificación de **calibración**.
- Interpretabilidad con **importancia por permutación**.

## Resultados
- La regla actual casi no ordena el riesgo (PR-AUC 0,03).
- Gradient boosting: PR-AUC 0,57 y ROC-AUC 0,96 en un mes nunca visto, con probabilidades bien calibradas.
- Decidiendo por pérdida esperada, el costo de junio baja un 24 % frente a la regla (~US$ 42.000 al año), con menos alertas y el 88 % del monto fraudulento detectado.
- Las alertas son el doble de precisas: 15 % contra 7 %.

## Archivos
| Archivo | Qué es |
|---|---|
| `analisis.ipynb` | Notebook ejecutado con el análisis completo, gráficos y conclusiones |
| `analisis.py` | El mismo análisis como script (celdas `# %%`) |
| `generar_datos.py` | Genera el dataset simulado en `data/` |
| `data/` | transacciones.csv.gz |

**Herramientas:** Python · pandas · scikit-learn · matplotlib

## Sobre los datos
Datos **simulados** con reglas de negocio realistas; no representan a ninguna empresa real. Para pasar a datos reales alcanza con reemplazar los archivos de `data/` por otros con el mismo esquema, por ejemplo el dataset de fraude con tarjetas de IEEE-CIS (Kaggle) o el histórico de transacciones con contracargos de la entidad.

## Cómo ejecutarlo
```bash
pip install pandas numpy scikit-learn scipy matplotlib
python generar_datos.py
jupyter notebook analisis.ipynb
```
