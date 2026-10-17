# Neurona de Izhikevich — TP1 Redes Neuronales 2026

Simulación numérica del modelo de neuronas de Izhikevich (2003): una neurona individual integrada con **Runge-Kutta 4** y, como parte opcional, una **red de 1000 neuronas** con ruido integrada con **Euler-Maruyama**.

Trabajo Práctico 1 del curso *Redes Neuronales 2026* — FaMAF, Universidad Nacional de Córdoba.

---

## El modelo

El modelo reduce la dinámica de Hodgkin-Huxley a dos ecuaciones:

```
dv/dt = g2·v² + g1·v + g0 − u + I(t)
du/dt = b·(c·v − u)

si v ≥ v+  →  v ← v−,   u ← u + Δu
```

| Símbolo | Qué representa |
|---|---|
| `v` | Potencial de membrana (mV) |
| `u` | Variable de recuperación (activación de K⁺ / inactivación de Na⁺) |
| `b` | Velocidad de recuperación de `u` |
| `c` | Sensibilidad de `u` a las fluctuaciones de `v` |
| `v−` | Valor de reseteo de `v` después de un disparo |
| `Δu` | Incremento de `u` después de un disparo |
| `I(t)` | Corriente de entrada externa |

Parámetros base: `g2 = 0.04`, `g1 = 5`, `g0 = 140`, `v+ = 30 mV`.

> Nota: la notación `b, c, v−, Δu`, corresponde a `a, b, c, d` del paper original.

---

## Contenido

### Parte 1 — Neurona individual

1. Integración con RK4 en `t ∈ [0, 200]` ms, paso `h = 0.1`, con `v(0) = −70` y `u(0) = c·v(0)`.
2. Corriente escalón: `I = 0` si `t < 10`, `I = 10` si `t ≥ 10`.
3. Reproducción de los 8 patrones de disparo de la Figura 2 del paper:

| Caso | Nombre | b | c | v− | Δu |
|---|---|---|---|---|---|
| RS | Regular spiking | 0.02 | 0.2 | −65 | 8 |
| IB | Intrinsically bursting | 0.02 | 0.2 | −55 | 4 |
| CH | Chattering | 0.02 | 0.2 | −50 | 2 |
| FS | Fast spiking | 0.1 | 0.2 | −65 | 2 |
| TC1 | Thalamo-cortical (despolarizado) | 0.02 | 0.25 | −65 | 0.05 |
| TC2 | Thalamo-cortical (hiperpolarizado) | 0.02 | 0.25 | −65 | 0.05 |
| RZ | Resonator | 0.1 | 0.26 | −65 | 2 |
| LTS | Low-threshold spiking | 0.02 | 0.25 | −65 | 2 |

### Parte 2 — Red de neuronas

- Traducción a Python del código MATLAB del paper.
- 800 neuronas excitatorias + 200 inhibitorias con parámetros heterogéneos (`r_i ~ U[0,1]`).
- Matriz de conexiones: `a_ij = 0.5·r_ij` (excitatorias) y `a_ij = −r_ij` (inhibitorias).
- Ruido gaussiano (σ = 5 excitatorias, σ = 2 inhibitorias) integrado con Euler-Maruyama.
- Reemplazo del escalón de Heaviside por `z(v) = (87 + v)/450 − 0.0193`.
- Reproducción de la Figura 3 del paper (raster plot de disparos).

---

## Estructura del repositorio

Autores (grupo 36): Mariel Anachuri, Camila Giolitto, Martín Robins.

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
