# Data Science & Analytics — Mateo Taylor

Cuatro proyectos de punta a punta: una **pregunta de negocio**, datos con problemas reales de calidad, el **análisis o el modelo** y una **recomendación con su impacto estimado**. Cada proyecto tiene un notebook ejecutado y un dashboard interactivo.

🔗 **Dashboards:** [taylorsportafolio.com/data](https://www.taylorsportafolio.com/data/)

| # | Proyecto | Perfil | Pregunta | Resultado |
|---|---|---|---|---|
| 01 | [Clientes, cohortes y LTV/CAC](projects/01-ecommerce-cohortes) | Data Analyst | ¿La tienda crece de forma sana? | El canal que más crece (Paid Social) tiene LTV/CAC de 3,0x y la peor retención |
| 02 | [Embudo de conversión y A/B test](projects/02-marketing-ab-test) | Data Analyst | ¿El checkout de una página vende más? | +5,1 pp de conversión (p < 0,001), casi todo en mobile: ~US$ 14.000 más por mes |
| 03 | [Detección de fraude](projects/03-fintech-fraude) | Data Scientist | ¿Cómo detectar más fraude con menos alertas? | −24 % de costo frente a la regla actual, con el 88 % del monto fraudulento detectado |
| 04 | [Pronóstico de demanda de bicis](projects/04-transporte-bicis) | Data Scientist | ¿Cuántas bicis se van a usar mañana, hora por hora? | −39 % de error frente a "igual que la semana pasada" (WAPE 16 % vs 26 %) |

## Cómo trabajo cada proyecto
1. **Pregunta de negocio:** arranco por la decisión que hay que tomar, no por el dataset.
2. **Datos y calidad:** encuentro duplicados, bots o categorías mal cargadas y documento cada corrección.
3. **Análisis o modelo:** siempre contra un baseline simple y con validación honesta (partición temporal, test fuera de muestra).
4. **Decisión e impacto:** cierro con una recomendación concreta, lo que vale en plata y su rango de incertidumbre.

## Sobre los datos
Los datasets son **simulados** con reglas de negocio realistas y errores de carga a propósito. No representan a ninguna empresa real. Cada proyecto incluye el script `generar_datos.py` que los crea, y el análisis está escrito para funcionar con datos reales que respeten el mismo esquema.

## Estructura
Esta carpeta vive dentro del repo del portfolio y se publica en [taylorsportafolio.com/data](https://www.taylorsportafolio.com/data/).
```
data/
  index.html               página índice de los proyectos
  ecommerce/ marketing/ fraude/ bicis/   dashboards (index.html generado desde src.html + data.json)
  assets/                  estilos y gráficos SVG propios, sin dependencias
  projects/
    01-ecommerce-cohortes/ generar_datos.py · analisis.py · analisis.ipynb · data/
    02-marketing-ab-test/
    03-fintech-fraude/
    04-transporte-bicis/
  tools/                   ejecutor de notebooks y armado de dashboards
```

## Cómo ejecutarlo
```bash
pip install pandas numpy scikit-learn scipy matplotlib
cd data/projects/01-ecommerce-cohortes
python generar_datos.py            # crea data/
jupyter notebook analisis.ipynb    # o ejecutá analisis.py
```
Para regenerar los notebooks y los dashboards sin Jupyter:
```bash
cd data
python tools/run_notebook.py projects/01-ecommerce-cohortes/analisis.py projects/01-ecommerce-cohortes/analisis.ipynb
python tools/build_dashboards.py
```

---
**English summary:** four end-to-end data projects (customer cohorts & LTV/CAC, funnel & A/B testing, fraud detection with cost-based decisions, and hourly bike-share demand forecasting), each with an executed notebook and an interactive dashboard. Data is simulated with realistic business rules; notebooks are written in Spanish and dashboards are bilingual (ES/EN).

Mateo Taylor · Licenciatura en Ciencia de Datos (UBA) · [taylorsportafolio.com](https://www.taylorsportafolio.com)
