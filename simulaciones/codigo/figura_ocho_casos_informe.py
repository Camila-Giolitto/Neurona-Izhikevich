import matplotlib
matplotlib.use('Agg')
import sys
OUT=sys.argv[1]
# --- celda 3
import numpy                   as np
import matplotlib.pyplot       as plt
import matplotlib.font_manager as fm

# --- celda 5
FUENTE = '/usr/share/fonts/opentype/inter/Inter-Regular.otf'

COLOR   = '#2a78d6'
NARANJA = '#eb6834'
FONDO   = '#fcfcfb'
GRILLA  = '#e4e3df'
TEXTO   = '#52514e'
TITULO  = '#0b0b0b'

fm.fontManager.addfont(FUENTE)

plt.rcParams.update(
                    {
                     'font.family'        : fm.FontProperties(fname = FUENTE).get_name(),
                     'figure.facecolor'   : '#ffffff',
                     'axes.facecolor'     : FONDO,
                     'axes.edgecolor'     : GRILLA,
                     'axes.grid'          : True,
                     'axes.axisbelow'     : True,
                     'axes.labelcolor'    : TEXTO,
                     'axes.titlecolor'    : TITULO,
                     'axes.titlelocation' : 'left',
                     'grid.color'         : GRILLA,
                     'grid.linewidth'     : 0.8,
                     'xtick.color'        : TEXTO,
                     'ytick.color'        : TEXTO,
                    }
                   )

# --- celda 7
def I0(t):
    if t < 10:
        return 0.0
    else:
        return 10.0

# --- celda 9
v_pico = 30.0

p = {
     'a'      : 0.02,
     'b'      : 0.2,
     'c'      : -65.0,
     'd'      : 2.0,
     'I'      : I0,
     'spikes' : [],
    }

v0 = -70.0
u0 = p['b'] * v0
x0 = np.array([v0, u0])

t0 = 0.0
tf = 100.0
h  = 0.1
k  = round((tf - t0) / h)

print('Condición inicial (v, u):', x0)
print('Pasos de integración    :', k)

# --- celda 11
def f_izh(x, t, p):
    v = x[0]
    u = x[1]

    dv = 0.04 * v**2 + 5 * v + 140 - u + p['I'](t)
    du = p['a'] * (p['b'] * v - u)

    return np.array([dv, du])


print('Derivadas en t = 0 :', f_izh(x0, 0, p))

# --- celda 13
def reseteo(x, t, p):
    if x[0] >= v_pico:
        p['spikes'].append(t)

        x[0] = p['c']
        x[1] = x[1] + p['d']

    return x

# --- celda 15
def rk4(f, x, t, h, p):
    k1 = f(x, t, p)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h, p)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h, p)
    k4 = f(x + h * k3, t + h, p)

    return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrador_ode(m, f, xa, t_ini, t_fin, k, p, c = lambda x, t, p: x):
    n = len(xa)
    h = (t_fin - t_ini) / k

    t = np.zeros(k + 1)
    w = np.zeros((n, k + 1))

    t[0]    = t_ini
    w[:, 0] = xa

    for j in range(k):
        t[j + 1]    = t[j] + h                         # avanza el tiempo
        w[:, j + 1] = m(f, w[:, j], t[j], h, p)        # un paso del método (rk4)
        w[:, j + 1] = c(w[:, j + 1], t[j + 1], p)      # función condicionante (reseteo)

    return t, w

# --- celda 17
p['spikes'] = []

t, w = integrador_ode(rk4, f_izh, x0, t0, tf, k, p, reseteo)

v      = w[0, :]
u      = w[1, :]
spikes = np.array(p['spikes'])

print('Disparos                :', len(spikes))
print('Tiempos de disparo (ms) :', spikes.round(1))

# --- celda 21
def I1(t):
    if t < 20:
        return 0.0
    else:
        return 10.0


def I2(t):
    if t < 20:
        return 0.0
    else:
        return 1.0


def I3(t):
    if t < 20:
        return 0.0
    elif t < 80:
        return -25.0
    else:
        return 0.0


def I4(t):
    if t < 40:
        return 0.0
    elif t < 180:
        return 0.2
    elif t < 181:
        return 2.2
    else:
        return 0.2

# --- celda 23
casos = {
         'RS'  : {'a': 0.02, 'b': 0.20, 'c': -65.0, 'd': 8.00, 'I': I1, 'v0': -65.0, 'tf': 200.0},
         'IB'  : {'a': 0.02, 'b': 0.20, 'c': -55.0, 'd': 4.00, 'I': I1, 'v0': -65.0, 'tf': 200.0},
         'CH'  : {'a': 0.02, 'b': 0.20, 'c': -50.0, 'd': 2.00, 'I': I1, 'v0': -65.0, 'tf': 200.0},
         'FS'  : {'a': 0.10, 'b': 0.20, 'c': -65.0, 'd': 2.00, 'I': I1, 'v0': -65.0, 'tf': 200.0},
         'TC1' : {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05, 'I': I2, 'v0': -63.0, 'tf': 200.0},
         'TC2' : {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05, 'I': I3, 'v0': -65.0, 'tf': 200.0},
         'RZ'  : {'a': 0.10, 'b': 0.26, 'c': -65.0, 'd': 2.00, 'I': I4, 'v0': -62.5, 'tf': 300.0},
         'LTS' : {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.00, 'I': I1, 'v0': -65.0, 'tf': 200.0},
        }

print('Casos:', list(casos.keys()))

# --- celda 27
resultados = {}

for nombre in casos:
    caso = casos[nombre]

    p_caso = {
              'a'      : caso['a'],
              'b'      : caso['b'],
              'c'      : caso['c'],
              'd'      : caso['d'],
              'I'      : caso['I'],
              'spikes' : [],
             }

    x0_caso = np.array([caso['v0'], caso['b'] * caso['v0']])
    k_caso  = round((caso['tf'] - t0) / h)

    t_caso, w_caso = integrador_ode(rk4, f_izh, x0_caso, t0, caso['tf'], k_caso, p_caso, reseteo)

    spikes_caso   = np.array(p_caso['spikes'])
    indices_spike = np.searchsorted(t_caso, spikes_caso)

    resultados[nombre] = {
                          't'         : t_caso,
                          'v'         : w_caso[0, :],
                          'u'         : w_caso[1, :],
                          'spikes'    : spikes_caso,
                          't_grafico' : np.insert(t_caso, indices_spike, spikes_caso),
                          'v_grafico' : np.insert(w_caso[0, :], indices_spike, v_pico),
                         }

    print(f'{nombre:4s}: {len(spikes_caso):3d} disparos')

# --- celda 37
fig = plt.figure(figsize = (7.5, 4.5), layout = 'constrained')
gs  = fig.add_gridspec(4, 4, height_ratios = [3, 1, 3, 1])

for i, nombre in enumerate(casos):
    fila    = 2 * (i // 4)
    columna = i % 4

    resultado = resultados[nombre]
    corriente = casos[nombre]['I']

    I_caso = np.zeros(len(resultado['t']))

    for j in range(len(resultado['t'])):
        I_caso[j] = corriente(resultado['t'][j])

    eje_v = fig.add_subplot(gs[fila, columna])
    eje_I = fig.add_subplot(gs[fila + 1, columna], sharex = eje_v)

    eje_v.plot(resultado['t_grafico'], resultado['v_grafico'], lw = 0.9, color = COLOR)
    eje_v.set_title(nombre, fontsize = 9)
    eje_v.set_ylim(-95, 35)
    eje_v.set_yticks(np.arange(-90, 31, 30))
    eje_v.tick_params(labelsize = 7, labelbottom = False)

    eje_I.plot(resultado['t'], I_caso, lw = 0.9, color = COLOR)
    eje_I.set_yticks([I_caso.min(), I_caso.max()])
    eje_I.tick_params(labelsize = 7)

    if columna == 0:
        eje_v.set_ylabel('v (mV)', fontsize = 8)
        eje_I.set_ylabel('I', fontsize = 8)
    else:
        eje_v.tick_params(labelleft = False)

    if fila == 2:
        eje_I.set_xlabel('t (ms)', fontsize = 8)

fig.savefig(OUT, dpi=300)