# Neurona de Izhikevich — TP1 Redes Neuronales 2026

Simulación numérica del modelo de neuronas de Izhikevich (2003): una neurona individual integrada con **Runge-Kutta 4** y, como parte opcional, una **red de 1000 neuronas** con ruido integrada con **Euler-Maruyama**.

Trabajo Práctico 1 del curso *Redes Neuronales 2026* — FaMAF, Universidad Nacional de Córdoba.

---

## El modelo

El modelo reduce la dinámica de Hodgkin-Huxley a dos ecuaciones (notación del informe y del código, la misma que el paper):

```
dv/dt = 0.04·v² + 5·v + 140 − u + I(t)
du/dt = a·(b·v − u)

si v ≥ 30 mV  →  v ← c,   u ← u + d
```

| Símbolo | Qué representa |
|---|---|
| `v` | Potencial de membrana (mV) |
| `u` | Variable de recuperación (activación de K⁺ / inactivación de Na⁺) |
| `I(t)` | Corriente de entrada externa |
| `a` | Escala temporal de `u` (velocidad de recuperación) |
| `b` | Sensibilidad de `u` a las fluctuaciones subumbrales de `v` |
| `c` | Valor de reseteo de `v` después de un disparo (mV) |
| `d` | Incremento de `u` después de un disparo |

> Nota: la consigna anterior (24/9) usaba `b, c, v−, Δu` en lugar de `a, b, c, d`. Esa notación sólo aparece en `versiones-previas/` y en `simulaciones/codigo/figuras_teoria.py`.

---

## Contenido

### Parte 1 — Neurona individual

1. Integración con RK4, paso `h = 0.1` ms, reseteo cuando `v ≥ 30` mV y condición inicial `u(0) = b·v(0)`.
2. Caso base en `t ∈ [0, 100]` ms con `v(0) = −70` mV (`u(0) = −14`) y escalón `I₀`: `I = 0` si `t < 10`, `I = 10` si `t ≥ 10`.
3. Reproducción de los 8 patrones de disparo de la Figura 2 del paper, simulados durante 200 ms (300 ms para RZ):

| Caso | Nombre | a | b | c (mV) | d | v(0) (mV) | Corriente (t en ms) |
|---|---|---|---|---|---|---|---|
| Base | Caso base | 0.02 | 0.2 | −65 | 2 | −70 | `I₀`: 0 (t < 10); 10 (t ≥ 10) |
| RS | Regular spiking | 0.02 | 0.2 | −65 | 8 | −65 | `I₁`: 0 (t < 20); 10 (t ≥ 20) |
| IB | Intrinsically bursting | 0.02 | 0.2 | −55 | 4 | −65 | `I₁` |
| CH | Chattering | 0.02 | 0.2 | −50 | 2 | −65 | `I₁` |
| FS | Fast spiking | 0.1 | 0.2 | −65 | 2 | −65 | `I₁` |
| TC1 | Thalamo-cortical (despolarizado) | 0.02 | 0.25 | −65 | 0.05 | −63 | `I₂`: 0 (t < 20); 1 (t ≥ 20) |
| TC2 | Thalamo-cortical (hiperpolarizado) | 0.02 | 0.25 | −65 | 0.05 | −65 | `I₃`: 0 (t < 20); −25 (20 ≤ t < 80); 0 (t ≥ 80) |
| RZ | Resonator | 0.1 | 0.26 | −65 | 2 | −62.5 | `I₄`: 0 (t < 40); 0.2 (40 ≤ t < 180); 2.2 (180 ≤ t < 181); 0.2 (t ≥ 181) |
| LTS | Low-threshold spiking | 0.02 | 0.25 | −65 | 2 | −65 | `I₁` |

4. Error numérico: el caso base se repite con `h = 0.2, 0.05, 0.01` ms, tomando `h = 0.01` como referencia.

### Parte 2 — Red de neuronas

- Traducción a Python del código MATLAB del paper.
- 800 neuronas excitatorias + 200 inhibitorias con parámetros heterogéneos (`r_i ~ U[0,1]`).
- Matriz de conexiones: `a_ij = 0.5·r_ij` (excitatorias) y `a_ij = −r_ij` (inhibitorias).
- Ruido gaussiano (σ = 5 excitatorias, σ = 2 inhibitorias) integrado con Euler-Maruyama.
- Reemplazo del escalón de Heaviside por `z(v) = (87 + v)/450 − 0.0193`.
- Reproducción de la Figura 3 del paper (raster plot de disparos).

---

## Estructura del repositorio

```
Neurona-Izhikevich/
├── README.md
├── requirements.txt
├── informe/
│   ├── grupo36_mariel_anachuri_camila_giolitto_martin_robins_tp1.pdf   # informe entregado
│   ├── latex/                 # fuente LaTeX del informe (main.tex + figuras/)
│   └── borrador-introduccion.docx
├── notebooks/
│   ├── TP1_Izhikevich.ipynb   # notebook final del grupo
│   └── versiones-previas/
├── simulaciones/
│   ├── codigo/                # módulos y scripts que generan las figuras
│   │   ├── izhikevich.py          # modelo, RK4/Euler, corrientes I1–I4, reseteo
│   │   ├── red_izhikevich.py      # red de 1000 neuronas (Euler-Maruyama)
│   │   ├── figuras.py             # regenera figs/ y resultados.json
│   │   ├── figuras_teoria.py, figuras_rk4.py        # figuras de las guías
│   │   ├── figura_ocho_casos_informe.py, figura_red_informe.py
│   │   └── resultados.json        # tiempos de disparo y métricas
│   ├── figs/                  # figuras generadas
│   └── tp1_izhikevich.ipynb   # notebook extendida (análisis complementario)
├── guias/                     # guías de estudio en HTML (abrir en el navegador)
└── versiones-previas/         # informe LaTeX anterior y material de la consigna del 24/9
```

### Cómo correr el código

```bash
pip install -r requirements.txt
cd simulaciones/codigo
python figuras.py              # regenera ../figs/*.png y resultados.json (~90 s)
python figuras_teoria.py       # figuras de teoría en ../figs/teoria/
python figuras_rk4.py
python figura_ocho_casos_informe.py salida.png   # figura 2x4 de los ocho tipos (300 dpi)
python figura_red_informe.py salida.png
```

Las notebooks se abren con `jupyter notebook notebooks/TP1_Izhikevich.ipynb`.

---

**Dependencias:** Python 3.10+, `numpy`, `scipy`, `matplotlib`, `jupyter`.

---

## Referencias

1. Izhikevich, E. M. (2003). *Simple model of spiking neurons*. IEEE Transactions on Neural Networks, 14(6), 1569–1572.
2. Izhikevich, E. M. (2006). *Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting*. MIT Press.
3. Hodgkin, A. L., & Huxley, A. F. (1952). *A quantitative description of membrane current and its application to conduction and excitation in nerve*. The Journal of Physiology, 117(4), 500–544.

---


Docentes: Francisco Tamarit, Juan Perotti, Tristán Osán, Franco Milana.
