# Clientes, cohortes y LTV/CAC de una tienda online
**Data Analyst · E-commerce** · Mateo Taylor

🔗 [Dashboard interactivo](https://www.taylorsportafolio.com/data/ecommerce/) · 📓 [Notebook](analisis.ipynb)

## Pregunta
La tienda crece en ventas, pero ¿está creciendo de forma sana? Qué canales traen clientes que vuelven a comprar, cuánto vale cada cliente frente a lo que cuesta conseguirlo y a qué segmentos dirigir cada acción.

## Qué hice
- Limpieza documentada: pedidos duplicados, cancelados y de prueba; regiones mal escritas o vacías.
- KPIs mensuales calculados con **SQL** sobre SQLite.
- Índice estacional sin tendencia (ajuste log-lineal).
- **Cohortes de retención** por mes de primera compra.
- **LTV a 12 meses, CAC y LTV/CAC** por canal de adquisición.
- Efecto del cupón de bienvenida en la recompra.
- **Segmentación RFM** con segmentos accionables.
- Categorías y tasa de devolución.

## Resultados
- Ingresos 2025 +71 % frente a 2024, pero el ticket se mantiene plano y solo el 38 % de los clientes activos son recurrentes.
- La retención no mejora: las cohortes de 2025 retienen igual que las de 2024 (≈10 % entre M1 y M6).
- Paid Social pasó del 48 % al 59 % de la inversión, pero solo el 22 % de sus clientes recompra y su LTV/CAC es 3,0x.
- Quienes entran con cupón recompran un 29 % (contra 43 %) y gastan un 31 % menos en su primer año.
- Campeones y Leales: 11 % de los clientes y 27 % de los ingresos.

## Archivos
| Archivo | Qué es |
|---|---|
| `analisis.ipynb` | Notebook ejecutado con el análisis completo, gráficos y conclusiones |
| `analisis.py` | El mismo análisis como script (celdas `# %%`) |
| `generar_datos.py` | Genera el dataset simulado en `data/` |
| `data/` | clientes.csv, pedidos.csv, items.csv, marketing.csv |

**Herramientas:** Python · pandas · SQL (SQLite) · matplotlib

## Sobre los datos
Datos **simulados** con reglas de negocio realistas; no representan a ninguna empresa real. Para pasar a datos reales alcanza con reemplazar los archivos de `data/` por otros con el mismo esquema, por ejemplo Olist Brazilian E-commerce (Kaggle) o el export de pedidos de una tienda Tiendanube / Shopify.

## Cómo ejecutarlo
```bash
pip install pandas numpy scikit-learn scipy matplotlib
python generar_datos.py
jupyter notebook analisis.ipynb
```
