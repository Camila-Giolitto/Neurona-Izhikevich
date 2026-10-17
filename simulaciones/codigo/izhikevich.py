"""TP1 Redes Neuronales 2026 - Modelo de Izhikevich (Primera parte).
Consigna actualizada (28/9/2026): notación del artículo original (a, b, c, d).
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

# ---------------------------------------------------------------- modelo (notación del artículo [1])
#   dv/dt = 0.04 v^2 + 5 v + 140 - u + I(t)
#   du/dt = a (b v - u)
#   si v >= 30 mV:  v <- c,  u <- u + d
# p = (a, b, c, d, I, v_pico)   con I una función de t
V_PICO = 30.0   # valor que usa el algoritmo para registrar el pico y resetear (NO es un umbral fisiológico)

def izhikevich(x, t, p):
    a, b, c, d, I, vp = p
    v, u = x
    return np.array([0.04*v**2 + 5.0*v + 140.0 - u + I(t),
                     a*(b*v - u)])

def reseteo(x, t, p):
    a, b, c, d, I, vp = p
    if x[0] >= vp:                       # spike registrado
        return np.array([c, x[1] + d])   # v <- c, u <- u + d
    return x

def parametros(a=0.02, b=0.2, c=-65.0, d=2.0, I=None, v_pico=V_PICO):
    return (a, b, c, d, I, v_pico)

# ---------------------------------------------------------------- corrientes (consigna actualizada)
def escalon(t0, I0, Iantes=0.0):
    return lambda t: Iantes if t < t0 else I0

def pulso(t0, t1, I0, Ifuera=0.0):
    return lambda t: I0 if t0 <= t < t1 else Ifuera

I_base = escalon(10, 10.0)              # parte 1.1: I = 10 para t >= 10 ms
I1 = escalon(20, 10.0)                  # RS, IB, CH, FS, LTS
I2 = escalon(20, 1.0)                   # TC1 (disparo tónico desde el reposo)
I3 = pulso(20, 80, -25.0)               # TC2 (hiperpolarización y rebound burst)
def I4(t):                              # RZ (corriente chica + pulso breve)
    if t < 40:   return 0.0
    if t < 180:  return 0.2
    if t < 181:  return 2.2
    return 0.2

#           a     b     c     d     I    v(0)
CASOS = {
    'RS':  (0.02, 0.20, -65, 8.00, I1, -65.0),
    'IB':  (0.02, 0.20, -55, 4.00, I1, -65.0),
    'CH':  (0.02, 0.20, -50, 2.00, I1, -65.0),
    'FS':  (0.10, 0.20, -65, 2.00, I1, -65.0),
    'TC1': (0.02, 0.25, -65, 0.05, I2, -63.0),
    'TC2': (0.02, 0.25, -65, 0.05, I3, -65.0),
    'RZ':  (0.10, 0.26, -65, 2.00, I4, -62.5),
    'LTS': (0.02, 0.25, -65, 2.00, I1, -65.0),
}
# duración de cada simulación (la consigna permite ajustarla para que se vea la dinámica)
DURACION = {'RS': 200, 'IB': 200, 'CH': 200, 'FS': 200, 'LTS': 200,
            'TC1': 200, 'TC2': 200, 'RZ': 300}

NOMBRES = {'RS': 'regular spiking (RS)', 'IB': 'intrinsically bursting (IB)',
           'CH': 'chattering (CH)', 'FS': 'fast spiking (FS)',
           'TC1': 'thalamo-cortical (TC1), disparo tónico', 'TC2': 'thalamo-cortical (TC2), rebound burst',
           'RZ': 'resonator (RZ)', 'LTS': 'low-threshold spiking (LTS)'}

# ---------------------------------------------------------------- análisis
def reposo(b, I=0.0):
    """Punto fijo estable (raíz menor) de 0.04 v^2 + (5-b) v + 140 + I = 0, u* = b v*."""
    disc = (5.0 - b)**2 - 4*0.04*(140.0 + I)
    if disc < 0:
        return None
    return (-(5.0 - b) - np.sqrt(disc))/(2*0.04)

def tiempos_disparo(t, w, c=None):
    """Índices j en que se aplicó el reseteo (v salta hacia abajo hasta c)."""
    return np.flatnonzero(np.diff(w[0]) < -15) + 1

def con_picos(t, w, vp=V_PICO):
    """Para graficar: inserta un punto (t_s, 30 mV) antes de cada reseteo, como
    hace Izhikevich ('all spikes were equalized at +30 mV')."""
    j = tiempos_disparo(t, w)
    tt = np.insert(t, j, t[j])
    vv = np.insert(w[0], j, vp)
    return tt, vv

def simular(p, v0, T=200.0, h=0.1, metodo=rk4):
    """RK4 (o Euler) + reseteo después de cada paso, con u(0) = b v(0)."""
    b = p[1]
    k = int(round(T/h))
    return integrador_ode(metodo, izhikevich, np.array([v0, b*v0]), 0.0, T, k, p, reseteo)

def p_caso(nombre):
    a, b, c, d, I, v0 = CASOS[nombre]
    return parametros(a=a, b=b, c=c, d=d, I=I)

def simular_caso(nombre, h=0.1, metodo=rk4, v0=None, T=None):
    v0 = CASOS[nombre][5] if v0 is None else v0
    return simular(p_caso(nombre), v0, T=T or DURACION[nombre], h=h, metodo=metodo)

# ---------------------------------------------------------------- RK4 con localización del evento
def integrador_eventos(f, xa, a, b, k, p):
    """RK4 + localización del cruce v = 30 por interpolación lineal dentro del paso.
    Devuelve la trayectoria en la grilla y los tiempos de disparo 'exactos'."""
    c, d, vp = p[2], p[3], p[5]
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
            xs = np.array([c, xs[1] + d])             # reseteo en el instante del cruce
            y = rk4(f, xs, ts, (1 - th)*h, p)         # completo el paso
        w[:, j+1] = y
    return t, w, np.array(spikes)
