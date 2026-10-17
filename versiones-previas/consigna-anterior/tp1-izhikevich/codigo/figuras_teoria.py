"""Figuras didácticas: qué representa cada letra del modelo de Izhikevich.
Notación del TP (entre paréntesis, la del paper):
  b (a), c (b), v_- (c), du (d), g2 g1 g0, v_+, I(t).
Genera figs/teoria/t*.png. Uso: python figuras_teoria.py"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs', 'teoria')
os.makedirs(OUT, exist_ok=True)
G2, G1, G0, VMAS = 0.04, 5.0, 140.0, 30.0

AZUL, NARANJA, VERDE, ROJO, GRIS, VIOLETA = '#1f5fa8', '#d9731a', '#2e8b57', '#c0392b', '#7a7f87', '#7b4fa0'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 130})


def reposo(c, I=0.0):
    """Raíz menor de g2 v^2 + (g1 - c) v + g0 + I = 0 (punto fijo estable)."""
    A, B, C = G2, G1 - c, G0 + I
    disc = B*B - 4*A*C
    if disc < 0:
        return None
    return (-B - np.sqrt(disc))/(2*A)


def simular(b=0.02, c=0.2, vm=-65.0, du=8.0, I=lambda t: 10.0*(t >= 10),
            T=200.0, h=0.1, v0=None, u0=None):
    """RK4 + reseteo. Devuelve t, v, u, I(t) y tiempos de spike.
    Para dibujar, el punto del disparo se guarda con v = v_+ (pico)."""
    if v0 is None:
        v0 = reposo(c)
    if u0 is None:
        u0 = c*v0
    f = lambda x, t: np.array([G2*x[0]**2 + G1*x[0] + G0 - x[1] + I(t), b*(c*x[0] - x[1])])
    n = int(round(T/h))
    t = np.arange(n+1)*h
    X = np.zeros((2, n+1)); X[:, 0] = (v0, u0)
    V = np.zeros(n+1); V[0] = v0
    sp = []
    x = np.array([v0, u0], float)
    for j in range(n):
        k1 = f(x, t[j]); k2 = f(x + 0.5*h*k1, t[j] + 0.5*h)
        k3 = f(x + 0.5*h*k2, t[j] + 0.5*h); k4 = f(x + h*k3, t[j] + h)
        x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6
        if x[0] >= VMAS:
            V[j+1] = VMAS
            X[:, j+1] = x            # u justo antes del salto (para dibujar)
            x = np.array([vm, x[1] + du])
            sp.append(t[j+1])
            continue
        V[j+1] = x[0]
        X[:, j+1] = x
    return t, V, X[1], np.array([I(s) for s in t]), np.array(sp)


def guardar(fig, nombre):
    fig.savefig(os.path.join(OUT, nombre), bbox_inches='tight')
    plt.close(fig)
    print('ok', nombre)


# ------------------------------------------------------------------ T1 anatomía de un disparo
def t1_anatomia():
    b, c, vm, du = 0.02, 0.2, -65.0, 8.0
    t, v, u, I, sp = simular(b, c, vm, du, T=120)
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})
    ax[0].plot(t, v, color=AZUL, lw=1.4)
    ax[0].axhline(VMAS, ls='--', color=ROJO, lw=1); ax[0].text(122, VMAS+1.5, r'$v_+=30$', color=ROJO, va='bottom')
    ax[0].axhline(vm, ls='--', color=VERDE, lw=1); ax[0].text(122, vm+1.5, r'$v_-=-65$', color=VERDE, va='bottom')
    ax[0].axhline(-70, ls=':', color=GRIS, lw=1); ax[0].text(122, -72, r'$v^*=-70$', color=GRIS, va='top')
    s0 = sp[0]
    ax[0].annotate('1) v cruza $v_+$:\nse cuenta un spike', xy=(s0, 30), xytext=(s0+36, 14),
                   arrowprops=dict(arrowstyle='->', color=ROJO), color=ROJO, fontsize=9)
    ax[0].annotate('2) reseteo:\n$v\\leftarrow v_-$', xy=(s0+0.2, vm), xytext=(s0+36, -42),
                   arrowprops=dict(arrowstyle='->', color=VERDE), color=VERDE, fontsize=9)
    ax[0].annotate('reposo', xy=(5, -70), xytext=(2, -55), arrowprops=dict(arrowstyle='->', color=GRIS),
                   color=GRIS, fontsize=9)
    ax[0].set_ylabel('v  [mV]'); ax[0].set_ylim(-80, 40); ax[0].set_xlim(-3, 140)
    ax[0].set_title('Anatomía de un disparo (RS: b=0.02, c=0.2, $v_-$=−65, Δu=8)')

    ax[1].plot(t, u, color=NARANJA, lw=1.4)
    for k, s in enumerate(sp[:3]):
        j = np.searchsorted(t, s)
        ax[1].annotate('', xy=(s, u[j]+du), xytext=(s, u[j]), arrowprops=dict(arrowstyle='->', color=VIOLETA, lw=1.5))
        if k == 0:
            ax[1].text(s+1.5, u[j]+du/2, r'3) $u\leftarrow u+\Delta u$', color=VIOLETA, fontsize=9, va='center')
    ax[1].set_ylabel('u'); ax[1].text(80, u.min()+1, 'entre spikes, u decae\ncon escala 1/b = 50 ms', fontsize=9, color=NARANJA)

    ax[2].plot(t, I, color=GRIS, lw=1.4); ax[2].set_ylabel('I(t)'); ax[2].set_xlabel('t  [ms]')
    ax[2].text(12, 6, 'escalón I = 10 en t = 10 ms', fontsize=9)
    fig.align_ylabels()
    guardar(fig, 't1_anatomia.png')


# ------------------------------------------------------------------ T2 parábola g2 g1 g0
def t2_parabola():
    v = np.linspace(-90, -20, 400)
    u = 0.2*(-70)   # u fijo en el valor del reposo
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for I, col in [(0, AZUL), (2, VERDE), (4, ROJO)]:
        dv = G2*v**2 + G1*v + G0 - u + I
        ax[0].plot(v, dv, color=col, label=f'I = {I}')
        r = np.roots([G2, G1, G0 - u + I])
        r = np.sort(r[np.isreal(r)].real)
        if len(r) == 2:
            ax[0].plot(r[0], 0, 'o', color=col); ax[0].plot(r[1], 0, 'o', mfc='white', color=col)
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].annotate('reposo\n(estable)', xy=(-70, 0), xytext=(-88, 3), arrowprops=dict(arrowstyle='->'), fontsize=9)
    ax[0].annotate('umbral\n(inestable)', xy=(-55, 0), xytext=(-50, 8), arrowprops=dict(arrowstyle='->'), fontsize=9)
    ax[0].annotate('', xy=(-80, -1.6), xytext=(-88, -1.6), arrowprops=dict(arrowstyle='->', color=GRIS))
    ax[0].annotate('', xy=(-74, -1.6), xytext=(-66, -1.6), arrowprops=dict(arrowstyle='->', color=GRIS))
    ax[0].text(-49, -3.7, 'pasado el umbral, v crece cada\nvez más rápido (∝ v²) y explota:\npor eso hace falta el corte $v_+$', fontsize=8.5, va='bottom')
    ax[0].set_xlabel('v  [mV]'); ax[0].set_ylabel(r'$\dot v = g_2v^2+g_1v+g_0-u+I$')
    ax[0].set_title('La parábola $g_2v^2+g_1v+g_0$ (u fijo = −14)')
    ax[0].set_ylim(-4, 14); ax[0].set_xlim(-92, -20); ax[0].legend(loc='upper left', fontsize=9)

    # derecha: el mismo efecto en el tiempo
    for I, col in [(3, VERDE), (6, ROJO)]:
        t, vv, uu, II, sp = simular(0.02, 0.2, -65, 8, I=lambda t, I=I: I*(t >= 10), T=150)
        ax[1].plot(t, vv, color=col, label=f'I = {I}')
    ax[1].set_xlabel('t  [ms]'); ax[1].set_ylabel('v  [mV]'); ax[1].legend(fontsize=9)
    ax[1].set_title('I=3: un spike y vuelve a un reposo nuevo\nI=6: ya no hay reposo, dispara siempre', fontsize=10)
    guardar(fig, 't2_parabola.png')


# ------------------------------------------------------------------ T3 b: escala temporal de u
def t3_b():
    fig, ax = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True)
    for k, (bb, col) in enumerate([(0.02, AZUL), (0.1, ROJO)]):
        t, v, u, I, sp = simular(bb, 0.2, -65, 2, T=200)
        ax[0, k].plot(t, v, color=col, lw=1); ax[1, k].plot(t, u, color=NARANJA, lw=1.3)
        ax[0, k].set_title(f'b = {bb}  →  1/b = {1/bb:.0f} ms   ({len(sp)} spikes)', color=col)
        ax[1, k].set_xlabel('t  [ms]')
    ax[0, 0].set_ylabel('v  [mV]'); ax[1, 0].set_ylabel('u')
    ax[1, 0].text(60, -9, 'u lenta: sus saltos se acumulan\n→ adaptación (los ISI crecen)', fontsize=9)
    ax[1, 1].text(60, -9, 'u rápida: cada salto se olvida\nantes del próximo → tren regular', fontsize=9)
    fig.suptitle('b (paper: a) = qué tan rápido u vuelve a c·v.   Fijos: c=0.2, $v_-$=−65, Δu=2, I=10', y=1.0)
    guardar(fig, 't3_b.png')


# ------------------------------------------------------------------ T4 c: sensibilidad de u a v
def t4_c():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    v = np.linspace(-80, -45, 300)
    ax[0].plot(v, G2*v**2 + G1*v + G0, color='k', lw=1.6, label=r'nulclina de v: $u=g_2v^2+g_1v+g_0$')
    for cc, col in [(0.2, AZUL), (0.25, VERDE), (0.26, ROJO)]:
        ax[0].plot(v, cc*v, color=col, lw=1.2, label=f'nulclina de u: u = {cc}·v')
        vs = reposo(cc)
        ax[0].plot(vs, cc*vs, 'o', color=col)
    ax[0].set_xlabel('v  [mV]'); ax[0].set_ylabel('u'); ax[0].set_ylim(-20, -8)
    ax[0].set_title('c = pendiente de la nulclina de u'); ax[0].legend(fontsize=8, loc='lower left')
    ax[0].text(-62, -9.5, 'más pendiente → reposo más\ndespolarizado y más cerca\ndel umbral (−70, −64.4, −62.5)', fontsize=8.5)

    pulso = lambda t: 2.0*(50 <= t < 51)
    for cc, col in [(0.2, AZUL), (0.26, ROJO)]:
        t, vv, uu, I, sp = simular(0.1, cc, -65, 2, I=pulso, T=150)
        ax[1].plot(t, vv - reposo(cc), color=col, label=f'c = {cc}')
    ax[1].set_xlabel('t  [ms]'); ax[1].set_ylabel(r'$v - v^*$  [mV]')
    ax[1].set_title('Respuesta a un pulso breve (b=0.1)'); ax[1].legend(fontsize=9)
    ax[1].text(80, 1.6, 'c grande: u "persigue" a v y\nla devuelve de más → oscilación\namortiguada (resonador, RZ)', fontsize=8.5)
    guardar(fig, 't4_c.png')


# ------------------------------------------------------------------ T5 v-: potencial de reseteo
def t5_vmenos():
    fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    for k, (vm, du, col, nom) in enumerate([(-65, 8, AZUL, 'RS'), (-55, 4, VERDE, 'IB'), (-50, 2, ROJO, 'CH')]):
        t, v, u, I, sp = simular(0.02, 0.2, vm, du, T=200)
        ax[k].plot(t, v, color=col, lw=1)
        ax[k].axhline(vm, ls='--', color=col, lw=0.8)
        ax[k].set_ylabel('v'); ax[k].set_ylim(-80, 40); ax[k].set_xlim(-5, 230)
        ax[k].text(205, vm, f'$v_-$={vm}', color=col, va='center', fontsize=9)
        ax[k].set_title(f'{nom}:  $v_-$ = {vm},  Δu = {du}', fontsize=10, loc='left', color=col)
    ax[2].set_xlabel('t  [ms]')
    ax[0].text(120, 5, 'reseteo hondo: después del\nspike v tiene que volver a subir', fontsize=8.5)
    ax[1].text(120, 5, 'reseteo cerca del umbral:\nráfaga inicial y después tónico', fontsize=8.5)
    ax[2].text(120, 5, 'reseteo por encima del umbral:\nráfagas repetidas', fontsize=8.5)
    fig.suptitle('$v_-$ (paper: c) = a dónde vuelve v después del spike.  b=0.02, c=0.2, I=10', y=1.0)
    guardar(fig, 't5_vmenos.png')


# ------------------------------------------------------------------ T6 du: salto de u por spike
def t6_du():
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True)
    for du, col in [(0.5, AZUL), (4, VERDE), (8, ROJO)]:
        t, v, u, I, sp = simular(0.02, 0.2, -65, du, T=200)
        ax[0].plot(sp, np.full_like(sp, du), '|', ms=14, mew=1.6, color=col)
        ax[1].plot(t, u, color=col, label=f'Δu = {du}  ({len(sp)} spikes)')
    ax[0].set_yticks([0.5, 4, 8]); ax[0].set_ylabel('Δu'); ax[0].set_ylim(-1, 10)
    ax[0].set_title('Tiempos de spike (raster) para distintos Δu')
    ax[1].set_ylabel('u'); ax[1].set_xlabel('t  [ms]'); ax[1].legend(fontsize=9)
    ax[1].set_title('cada spike sube u en Δu, y u frena a v (término −u en dv/dt):\nΔu grande = más adaptación, menos spikes', fontsize=9.5)
    ax[1].legend(fontsize=8.5, loc='lower right')
    fig.suptitle('Δu (paper: d) = cuánta "fatiga" deja cada spike.  b=0.02, c=0.2, $v_-$=−65, I=10', y=1.0)
    guardar(fig, 't6_du.png')


# ------------------------------------------------------------------ T7 plano de fase: todo junto
def t7_fase():
    b, c, vm, du, I0 = 0.02, 0.2, -65.0, 8.0, 10.0
    t, v, u, I, sp = simular(b, c, vm, du, I=lambda t: I0, T=200, v0=-70, u0=-14)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    vv = np.linspace(-80, 32, 400)
    ax.plot(vv, G2*vv**2 + G1*vv + G0 + I0, color='k', lw=1.5, label=r'nulclina de v ($\dot v=0$), con I=10')
    ax.plot(vv, G2*vv**2 + G1*vv + G0, color='k', lw=1, ls=':', label='nulclina de v con I=0')
    ax.plot(vv, c*vv, color=NARANJA, lw=1.5, label=r'nulclina de u ($\dot u=0$): u = c·v')
    ax.axvline(VMAS, color=ROJO, ls='--', lw=1); ax.axvline(vm, color=VERDE, ls='--', lw=1)
    ax.text(VMAS-1, 18, '$v_+$', color=ROJO, ha='right'); ax.text(vm+0.5, 18, '$v_-$', color=VERDE)
    # trayectoria, sin dibujar los saltos de reseteo como líneas
    cortes = np.where(np.diff(v) < -50)[0] + 1
    for seg in np.split(np.arange(len(v)), cortes):
        ax.plot(v[seg], u[seg], color=AZUL, lw=1.1)
    for s in sp[:4]:
        j = np.searchsorted(t, s)
        ax.annotate('', xy=(vm, u[j]+du), xytext=(VMAS, u[j]),
                    arrowprops=dict(arrowstyle='->', color=VIOLETA, lw=1, ls='--'))
    ax.text(-20, -2, 'reseteo: salto a la izquierda\n($v_-$) y hacia arriba (Δu)', color=VIOLETA, fontsize=9)
    ax.plot(-70, -14, 'o', color=GRIS); ax.text(-68, -17.5, 'inicio (reposo con I=0)', fontsize=8.5, color=GRIS)
    ax.set_xlim(-82, 35); ax.set_ylim(-20, 22)
    ax.set_xlabel('v  [mV]'); ax.set_ylabel('u'); ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title('Plano de fase (RS, I=10): con I=10 las nulclinas no se cortan → no hay reposo')
    guardar(fig, 't7_fase.png')


# ------------------------------------------------------------------ T8 I: corriente de entrada
def t8_I():
    fig, ax = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    for k, (I0, col) in enumerate([(3, GRIS), (5, AZUL), (10, VERDE), (20, ROJO)]):
        t, v, u, I, sp = simular(0.02, 0.2, -65, 8, I=lambda t, I0=I0: I0*(t >= 10), T=300)
        ax[k].plot(t, v, color=col, lw=1)
        f = 1000/np.mean(np.diff(sp[-3:])) if len(sp) >= 3 else 0
        txt = f'frecuencia final ≈ {f:.0f} Hz' if f else 'después vuelve al reposo'
        ax[k].set_title(f'I = {I0}:  {len(sp)} spike{"s" if len(sp) != 1 else ""}, {txt}', loc='left', fontsize=10, color=col)
        ax[k].set_ylabel('v'); ax[k].set_ylim(-85, 40)
    ax[3].set_xlabel('t  [ms]')
    fig.suptitle('I(t) = corriente que entra a la neurona (RS). Más I → más rápido se recarga v', y=1.0)
    guardar(fig, 't8_I.png')


if __name__ == '__main__':
    t1_anatomia(); t2_parabola(); t3_b(); t4_c(); t5_vmenos(); t6_du(); t7_fase(); t8_I()
