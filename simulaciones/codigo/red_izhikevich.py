"""TP1 Redes Neuronales 2026 - Segunda parte (opcional): red de 1000 neuronas de Izhikevich.
Consigna actualizada (28/9/2026):
  1) traducción literal del Matlab del artículo (acoplamiento por eventos de spike),
  2) raster tipo figura 3 con semilla reportada,
  3) comparación con / sin autoconexiones (s_ii = 0),
  extras opcionales: Euler-Maruyama con ruido blanco continuo y acoplamiento continuo con z(v)."""
import numpy as np

SEMILLA = 2026          # semilla reportada en el informe

# ---------------------------------------------------------------- parámetros de la red
def parametros_red(Ne=800, Ni=200, seed=SEMILLA, autoconexiones=True):
    """Mismo orden de sorteos que el Matlab: re, ri, después S.
    Con la misma semilla, las variantes con y sin diagonal comparten r_i, S y ruido."""
    rng = np.random.default_rng(seed)
    re, ri = rng.random(Ne), rng.random(Ni)
    a = np.r_[0.02*np.ones(Ne), 0.02 + 0.08*ri]
    b = np.r_[0.2*np.ones(Ne),  0.25 - 0.05*ri]
    c = np.r_[-65 + 15*re**2,   -65*np.ones(Ni)]       # -65 + 15 r^2 (paper)
    d = np.r_[8 - 6*re**2,       2*np.ones(Ni)]
    N = Ne + Ni
    S = np.hstack([0.5*rng.random((N, Ne)), -rng.random((N, Ni))])   # s_ij: de j (pre) a i (post)
    if not autoconexiones:
        np.fill_diagonal(S, 0.0)
    return dict(Ne=Ne, Ni=Ni, N=N, a=a, b=b, c=c, d=d, S=S, rng=rng)

def ruido_talamico(p):
    """I_ext: 5 xi para excitatorias, 2 xi para inhibitorias, xi ~ N(0,1), renovado cada 1 ms."""
    return np.r_[5*p['rng'].standard_normal(p['Ne']), 2*p['rng'].standard_normal(p['Ni'])]

# ---------------------------------------------------------------- 1) traducción literal del Matlab
def red_literal(p, T=1000):
    """Un paso = 1 ms. Detección (v >= 30) y reseteo al principio del paso,
    entrada sináptica por pulsos I_syn = sum_{j en F} s_ij, y dos medios pasos de Euler para v."""
    v = -65.0*np.ones(p['N']); u = p['b']*v
    firings = []
    for t in range(T):
        I = ruido_talamico(p)
        fired = np.flatnonzero(v >= 30)
        firings += [(t, i) for i in fired]
        v[fired] = p['c'][fired]
        u[fired] += p['d'][fired]
        I += p['S'][:, fired].sum(axis=1)                    # acoplamiento por eventos de spike
        v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)             # "for numerical stability"
        v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)
        u += p['a']*(p['b']*v - u)
    return np.array(firings, dtype=float).reshape(-1, 2)

# ---------------------------------------------------------------- extra: Euler-Maruyama (ruido blanco continuo)
def red_euler_maruyama(p, T=1000.0, h=0.1):
    """dV = (0.04V^2+5V+140-U) dt + sigma dW entre spikes; saltos sinápticos y reseteo por separado.
    El incremento de Wiener en un paso h tiene desvío sqrt(h): Y_{n+1} = Y_n + f h + sigma sqrt(h) xi."""
    sigma = np.r_[5*np.ones(p['Ne']), 2*np.ones(p['Ni'])]
    v = -65.0*np.ones(p['N']); u = p['b']*v
    k = int(round(T/h))
    firings = []
    for n in range(k):
        fired = np.flatnonzero(v >= 30)
        firings += [(n*h, i) for i in fired]
        v[fired] = p['c'][fired]
        u[fired] += p['d'][fired]
        v += p['S'][:, fired].sum(axis=1)                    # salto sináptico (misma amplitud que el original)
        fv = 0.04*v**2 + 5*v + 140 - u
        fu = p['a']*(p['b']*v - u)
        v = v + fv*h + sigma*np.sqrt(h)*p['rng'].standard_normal(p['N'])
        u = u + fu*h
    return np.array(firings, dtype=float).reshape(-1, 2)

# ---------------------------------------------------------------- extra: acoplamiento continuo con z(v)
def z(v):
    """z(v) = (87+v)/450 - 0.0193, con v recortado en 30 mV (v puede pasarse de 30 dentro del paso)."""
    return (87.0 + np.minimum(v, 30.0))/450.0 - 0.0193

def red_z(p, T=1000):
    """Igual que red_literal, pero la entrada sináptica es continua:
        I_syn,i(t) = sum_j s_ij z(v_j(t))      (en lugar de sum_{j en F} s_ij).
    Es un modelo distinto del de la figura 3: experimento comparativo, no reproducción."""
    v = -65.0*np.ones(p['N']); u = p['b']*v
    firings = []
    for t in range(T):
        I = ruido_talamico(p)
        I += p['S'] @ z(v)                                   # acoplamiento continuo (v antes del reseteo)
        fired = np.flatnonzero(v >= 30)
        firings += [(t, i) for i in fired]
        v[fired] = p['c'][fired]
        u[fired] += p['d'][fired]
        v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)
        v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)
        u += p['a']*(p['b']*v - u)
    return np.array(firings, dtype=float).reshape(-1, 2)

# ---------------------------------------------------------------- métricas
def metricas(firings, T, N=1000, Ne=800, desde=0.0):
    """Total de spikes y tasas medias (Hz por neurona) en [desde, T)."""
    m = firings[:, 0] >= desde
    t, i = firings[m, 0], firings[m, 1]
    dur = (T - desde)*1e-3
    return dict(spikes=int(t.size), tasa_Hz=float(t.size/(N*dur)),
                tasa_exc_Hz=float((i < Ne).sum()/(Ne*dur)), tasa_inh_Hz=float((i >= Ne).sum()/((N-Ne)*dur)))

def actividad_poblacional(firings, T, N=1000, bin_ms=1.0):
    bins = np.arange(0, T + bin_ms, bin_ms)
    cuentas, _ = np.histogram(firings[:, 0], bins)
    return bins[:-1], cuentas/(N*bin_ms*1e-3)               # Hz por neurona

def espectro(firings, T, N=1000, descartar=100.0):
    from scipy.signal import welch
    tb, tasa = actividad_poblacional(firings, T, N)
    x = tasa[tb >= descartar]
    f, P = welch(x - x.mean(), fs=1000, nperseg=min(512, x.size))
    return f, P
