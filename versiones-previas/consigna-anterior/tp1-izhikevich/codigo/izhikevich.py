"""TP1 Redes Neuronales 2026 - Modelo de Izhikevich (Primera parte).
Integradores (guía 6), modelo, corrientes de entrada y utilidades."""
import numpy as np

# ---------------------------------------------------------------- integradores (guía 6)
def euler(f, x, t, h, p):
    return x + h*f(x, t, p)

def rk4(f, x, t, h, p):
    k1 = f(x, t, p)
    k2 = f(x + 0.5*h*k1, t + 0.5*h, p)
    k3 = f(x + 0.5*h*k2, t + 0.5*h, p)
    k4 = f(x + h*k3, t + h, p)
    return x + h*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

def integrador_ode(m, f, xa, a, b, k, p, c=lambda x, t, p: x):
    """Integra dx/dt = f(x,t,p) en [a,b] con k pasos del método m.
    c(x,t,p) es la 'función condicionante' que se aplica después de cada paso:
    acá la usamos para implementar el reseteo (el 'truquito' del disparo)."""
    assert k > 0
    n = len(xa)
    h = (b - a)/k
    w = np.zeros((n, k+1))
    t = np.zeros(k+1)
    w[:, 0] = xa
    t[0] = a
    for j in range(k):
        t[j+1] = t[j] + h
        w[:, j+1] = m(f, w[:, j], t[j], h, p)
        w[:, j+1] = c(w[:, j+1], t[j+1], p)
    return t, w

# ---------------------------------------------------------------- modelo
# p = (g2, g1, g0, v_menos, v_mas, b, c, du, I)   con I una función de t
def izhikevich(x, t, p):
    g2, g1, g0, vm, vp, b, c, du, I = p
    v, u = x
    return np.array([g2*v**2 + g1*v + g0 - u + I(t),
                     b*(c*v - u)])

def reseteo(x, t, p):
    g2, g1, g0, vm, vp, b, c, du, I = p
    if x[0] >= vp:                       # disparo
        return np.array([vm, x[1] + du])  # v <- v-, u <- u + du
    return x

def parametros(b=0.02, c=0.2, v_menos=-65.0, du=2.0, I=None,
               g2=0.04, g1=5.0, g0=140.0, v_mas=30.0):
    return (g2, g1, g0, v_menos, v_mas, b, c, du, I)

# ---------------------------------------------------------------- corrientes
def escalon(t0, I0, Iantes=0.0):
    return lambda t: Iantes if t < t0 else I0

I_base = escalon(10, 10.0)              # parte 1.1
I1 = escalon(25, 10.0)                  # RS, IB, CH, FS, LTS
I2 = escalon(150, 2.5)                  # TC1 (despolarización desde el reposo)
I3 = escalon(100, 0.0, Iantes=-15.0)    # TC2 (hiperpolarización y rebote)
def I4(t):                              # RZ (pulso breve sobre un escalón chico)
    if t < 10:   return 0.0
    if t < 100:  return 0.2
    if t < 101:  return 10.0
    return 0.2

CASOS = {  #  b     c     v-    du    I
    'RS':  (0.02, 0.20, -65, 8.00, I1),
    'IB':  (0.02, 0.20, -55, 4.00, I1),
    'CH':  (0.02, 0.20, -50, 2.00, I1),
    'FS':  (0.10, 0.20, -65, 2.00, I1),
    'TC1': (0.02, 0.25, -65, 0.05, I2),
    'TC2': (0.02, 0.25, -65, 0.05, I3),
    'RZ':  (0.10, 0.26, -65, 2.00, I4),
    'LTS': (0.02, 0.25, -65, 2.00, I1),
}

NOMBRES = {'RS': 'regular spiking (RS)', 'IB': 'intrinsically bursting (IB)',
           'CH': 'chattering (CH)', 'FS': 'fast spiking (FS)',
           'TC1': 'thalamo-cortical (TC), despolarizado', 'TC2': 'thalamo-cortical (TC), rebote',
           'RZ': 'resonator (RZ)', 'LTS': 'low-threshold spiking (LTS)'}

# ---------------------------------------------------------------- análisis
def reposo(c, I=0.0, g2=0.04, g1=5.0, g0=140.0):
    """Punto fijo estable (raíz menor) de g2 v^2 + (g1-c) v + g0 + I = 0, u* = c v*."""
    disc = (g1 - c)**2 - 4*g2*(g0 + I)
    if disc < 0:
        return None
    v = (-(g1 - c) - np.sqrt(disc))/(2*g2)
    return v

def tiempos_disparo(t, w, vm):
    """Índices j en que se aplicó el reseteo (v salta hacia abajo hasta v-)."""
    j = np.flatnonzero(np.diff(w[0]) < -15) + 1
    return j

def con_picos(t, w, vp=30.0):
    """Para graficar: inserta un punto (t_s, v+) antes de cada reseteo, como
    hace Izhikevich ('all spikes were equalized at +30 mV')."""
    j = tiempos_disparo(t, w, None)
    tt = np.insert(t, j, t[j])
    vv = np.insert(w[0], j, vp)
    return tt, vv

def simular(caso_o_param, v0=None, T=200.0, h=0.1, metodo=rk4):
    p = caso_o_param
    c = p[6]
    if v0 is None:
        v0 = reposo(c)
    k = int(round(T/h))
    return integrador_ode(metodo, izhikevich, np.array([v0, c*v0]), 0.0, T, k, p, reseteo)

def p_caso(nombre):
    b, c, vm, du, I = CASOS[nombre]
    return parametros(b=b, c=c, v_menos=vm, du=du, I=I)

# ---------------------------------------------------------------- RK4 con localización del evento
def integrador_eventos(f, xa, a, b, k, p):
    """RK4 + localización del cruce v = v+ por interpolación lineal dentro del paso.
    Devuelve la trayectoria en la grilla y los tiempos de disparo 'exactos'."""
    vp, vm, du = p[4], p[3], p[7]
    h = (b - a)/k
    t = a + h*np.arange(k+1)
    w = np.zeros((len(xa), k+1)); w[:, 0] = xa
    spikes = []
    for j in range(k):
        x = w[:, j]
        y = rk4(f, x, t[j], h, p)
        if y[0] >= vp:
            th = (vp - x[0])/(y[0] - x[0])           # fracción del paso donde cruza
            xs = rk4(f, x, t[j], th*h, p)             # avanzo hasta el cruce
            ts = t[j] + th*h
            spikes.append(ts)
            xs = np.array([vm, xs[1] + du])           # reseteo en el instante del cruce
            y = rk4(f, xs, ts, (1 - th)*h, p)         # completo el paso
        w[:, j+1] = y
    return t, w, np.array(spikes)
