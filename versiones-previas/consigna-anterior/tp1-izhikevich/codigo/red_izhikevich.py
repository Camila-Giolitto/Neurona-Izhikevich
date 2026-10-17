"""TP1 Redes Neuronales 2026 - Segunda parte (opcional): red de Izhikevich
integrada con Euler-Maruyama. Activación Θ (Heaviside) o z(v)."""
import numpy as np

# ---------------------------------------------------------------- 3.1 traducción literal del MATLAB del paper
def red_matlab_literal(Ne=800, Ni=200, T=1000, seed=0):
    rng = np.random.default_rng(seed)
    re, ri = rng.random(Ne), rng.random(Ni)
    a = np.r_[0.02*np.ones(Ne), 0.02 + 0.08*ri]
    b = np.r_[0.2*np.ones(Ne), 0.25 - 0.05*ri]
    c = np.r_[-65 + 15*re**2, -65*np.ones(Ni)]
    d = np.r_[8 - 6*re**2, 2*np.ones(Ni)]
    S = np.hstack([0.5*rng.random((Ne+Ni, Ne)), -rng.random((Ne+Ni, Ni))])
    v = -65*np.ones(Ne+Ni); u = b*v
    firings = []
    for t in range(1, T+1):                         # 1 ms por iteración
        I = np.r_[5*rng.standard_normal(Ne), 2*rng.standard_normal(Ni)]  # input talámico
        fired = np.flatnonzero(v >= 30)
        firings += [(t, i) for i in fired]
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]
        I = I + S[:, fired].sum(axis=1)
        v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # dos medios pasos de 0.5 ms
        v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # "for numerical stability"
        u = u + a*(b*v - u)
    return np.array(firings)

# ---------------------------------------------------------------- 3.2 versión "teórica" (Euler-Maruyama)
def parametros_red(ne=800, ni=200, seed=None):
    """b, c, v-, du por neurona (heterogeneidad vía r_i), matriz a_ij y ruido w_i."""
    rng = np.random.default_rng(seed)
    n = ne + ni
    re, ri = rng.random(ne), rng.random(ni)
    b       = np.concatenate([0.02*np.ones(ne), 0.02 + 0.08*ri])   # RS..CH | FS..LTS
    c       = np.concatenate([0.2*np.ones(ne),  0.25 - 0.05*ri])
    v_menos = np.concatenate([-65 + 15*re**2,   -65*np.ones(ni)])
    du      = np.concatenate([8 - 6*re**2,       2*np.ones(ni)])
    A = np.hstack([0.5*rng.random((n, ne)), -rng.random((n, ni))])  # a_ij
    w = np.concatenate([5*np.ones(ne), 2*np.ones(ni)])              # g_ii
    return dict(ne=ne, ni=ni, n=n, b=b, c=c, v_menos=v_menos, du=du,
                A=A, w=w, g2=0.04, g1=5.0, g0=140.0, v_mas=30.0)

def theta(v, p):
    """Heaviside sobre el disparo: 1 si la neurona j alcanzó v+."""
    return (v >= p['v_mas']).astype(float)

def z(v, p):
    """Reemplazo del enunciado z(v) = (87+v)/450 - 0.0193 (v recortado en v+)."""
    return (87.0 + np.minimum(v, p['v_mas']))/450.0 - 0.0193

def drift(v, u, I_syn, p):
    fv = p['g2']*v**2 + p['g1']*v + p['g0'] - u + I_syn
    fu = p['b']*(p['c']*v - u)
    return fv, fu

def paso_euler_maruyama(v, u, I_syn, tau, p, rng):
    """Y(t+τ) = Y(t) + f τ + g √τ N(0,1); sólo las V tienen difusión."""
    fv, fu = drift(v, u, I_syn, p)
    N = rng.standard_normal(p['n'])
    return v + fv*tau + p['w']*np.sqrt(tau)*N, u + fu*tau

def reseteo(v, u, p):
    disparo = v >= p['v_mas']
    v = np.where(disparo, p['v_menos'], v)
    u = np.where(disparo, u + p['du'], u)
    return v, u, disparo

def simular_red(p, T=1000.0, tau=0.5, activacion=theta, seed=None, registrar=0):
    rng = np.random.default_rng(seed)
    k = int(round(T/tau))
    v = -65.0*np.ones(p['n']); u = p['c']*v
    t_spk, i_spk = [], []
    t_reg = np.arange(k+1)*tau
    v_reg = np.zeros(k+1); v_reg[0] = v[registrar]
    for j in range(k):
        t = j*tau
        act = activacion(v, p)                 # con v ANTES del reseteo
        I_syn = p['A'] @ act
        if activacion is theta:
            I_syn = I_syn/tau                  # cada spike transfiere 'a_ij' mV, como en el paper
        v, u, disparo = reseteo(v, u, p)
        idx = np.flatnonzero(disparo)
        t_spk.append(np.full(idx.size, t)); i_spk.append(idx)
        if disparo[registrar]:
            v_reg[j] = p['v_mas']
        v, u = paso_euler_maruyama(v, u, I_syn, tau, p, rng)
        v_reg[j+1] = min(v[registrar], p['v_mas'])
    return np.concatenate(t_spk), np.concatenate(i_spk), t_reg, v_reg

def actividad_poblacional(t_spk, T, n, bin_ms=1.0):
    bins = np.arange(0, T + bin_ms, bin_ms)
    cuentas, _ = np.histogram(t_spk, bins)
    tasa = cuentas/(n*bin_ms*1e-3)             # Hz por neurona
    x = tasa - tasa.mean()
    frec = np.fft.rfftfreq(x.size, d=bin_ms*1e-3)
    pot = np.abs(np.fft.rfft(x))**2
    return bins[:-1], tasa, frec, pot
