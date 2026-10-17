"""Figuras didácticas: el integrador RK4, el 'truquito' del disparo y el caso base (TP1, 1.1).
Usa las funciones de la guía 6 que están en izhikevich.py.
Genera figs/teoria/r*.png. Uso: python figuras_rk4.py"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from izhikevich import euler, rk4, integrador_ode, izhikevich, reseteo, parametros, I_base

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs', 'teoria')
os.makedirs(OUT, exist_ok=True)
AZUL, NARANJA, VERDE, ROJO, GRIS, VIOLETA, INK = '#1f5fa8', '#d9731a', '#2e8b57', '#c0392b', '#7a7f87', '#7b4fa0', '#17212c'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 130})
P = parametros(I=I_base)            # caso base: b=0.02, c=0.2, v-=-65, du=2, I=10 desde t=10
X0 = np.array([-70.0, 0.2*-70.0])   # v(0)=-70, u(0)=c v(0)


def guardar(fig, nombre):
    fig.savefig(os.path.join(OUT, nombre), bbox_inches='tight'); plt.close(fig); print('ok', nombre)


def correr(m, h, T=200.0, x0=X0, p=P, con_reset=True):
    """Igual que integrador_ode, pero guarda también la propuesta antes del reseteo."""
    k = int(round(T/h)); t = np.zeros(k+1)
    w = np.zeros((2, k+1)); w[:, 0] = x0; sp = []; prop = []
    for j in range(k):
        t[j+1] = t[j] + h          # el tiempo se acumula igual que en integrador_ode
        xn = m(izhikevich, w[:, j], t[j], h, p)
        if con_reset and xn[0] >= p[4]:
            sp.append(t[j+1]); prop.append(xn.copy())
            xn = reseteo(xn, t[j+1], p)
        w[:, j+1] = xn
    return t, w, np.array(sp), prop


# ------------------------------------------------------------------ R1 geometría de un paso RK4
def r1_paso():
    # paso grande a propósito, en la subida del primer spike, para que se vean las pendientes
    t, w, _, _ = correr(rk4, 0.001, T=13.0)
    tn, xn = 12.0, w[:, 12000]
    h = 1.0
    f = lambda x, s: izhikevich(x, s, P)
    k1 = f(xn, tn); x2 = xn + 0.5*h*k1
    k2 = f(x2, tn + h/2); x3 = xn + 0.5*h*k2
    k3 = f(x3, tn + h/2); x4 = xn + h*k3
    k4 = f(x4, tn + h)
    x_rk4 = xn + h*(k1 + 2*k2 + 2*k3 + k4)/6
    x_eu = xn + h*k1
    Pc = parametros(I=lambda s: 10.0)   # para t >= 10 la corriente es 10
    tr, wr, _, _ = correr(rk4, 0.0005, T=h, x0=xn, p=Pc, con_reset=False)  # 'exacto' desde tn
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(tn + tr[tr <= h], wr[0, tr <= h], color=INK, lw=1.8, label='solución "exacta" (h muy chico)')
    seg = 0.22
    pts = [(tn, xn[0], k1[0], 'k₁', AZUL), (tn+h/2, x2[0], k2[0], 'k₂', VERDE),
           (tn+h/2, x3[0], k3[0], 'k₃', NARANJA), (tn+h, x4[0], k4[0], 'k₄', ROJO)]
    for s, v, kk, nom, col in pts:
        ax.plot(s, v, 'o', color=col, ms=6, zorder=4)
        ax.plot([s-seg, s+seg], [v-seg*kk, v+seg*kk], color=col, lw=3, alpha=0.8)
        ax.annotate(f'{nom} = {kk:.1f}', xy=(s, v), xytext=(8, -14), textcoords='offset points', color=col, fontsize=9.5)
    ax.plot(tn+h, x_eu[0], 's', color=GRIS, ms=8, mfc='white', mew=1.8, label=f'Euler: v = {x_eu[0]:.1f}')
    ax.plot(tn+h, x_rk4[0], 'D', color=VIOLETA, ms=8, label=f'RK4: v = {x_rk4[0]:.1f}')
    ax.plot(tn+h, wr[0, -1], '*', color=INK, ms=12, label=f'exacto: v = {wr[0, -1]:.1f}')
    ax.axvline(tn+h/2, color=GRIS, ls=':', lw=1)
    ax.set_xticks([tn, tn+h/2, tn+h]); ax.set_xticklabels(['$t_n$', '$t_n + h/2$', '$t_n + h$'])
    ax.set_ylabel('v  [mV]'); ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'Un paso de RK4 con h = {h} ms (exagerado), en la subida del primer spike')
    ax.text(tn+h*0.64, -59.5, 'RK4 avanza con el promedio pesado\n(k₁ + 2k₂ + 2k₃ + k₄)/6;\nEuler sólo con k₁', fontsize=9, color=VIOLETA, va='bottom')
    ax.set_ylim(-60, None)
    guardar(fig, 'r1_paso_rk4.png')


# ------------------------------------------------------------------ R2 orden del método
def r2_orden():
    hs = np.array([0.4, 0.2, 0.1, 0.05, 0.025, 0.0125])
    # I = 10 constante desde t = 0 (sin el escalón, que es una discontinuidad y baja el orden)
    Pc = parametros(I=lambda t: 10.0)
    # (a) tramo suave: error en v(t=3) antes del 1er spike (t1 ≈ 3.5)
    Tf = 3.0
    ref = correr(rk4, 0.0005, T=Tf, p=Pc)[1][0, -1]
    eE, eR = [], []
    for h in hs:
        eE.append(abs(correr(euler, h, T=Tf, p=Pc)[1][0, -1] - ref))
        eR.append(abs(correr(rk4, h, T=Tf, p=Pc)[1][0, -1] - ref))
    # (b) tiempos de disparo en 100 ms: error máximo
    spr = correr(rk4, 0.0005, T=100, p=Pc)[2]
    sE, sR = [], []
    for h in hs:
        a = correr(euler, h, T=100, p=Pc)[2]; b = correr(rk4, h, T=100, p=Pc)[2]
        n = min(len(a), len(spr)); sE.append(np.max(np.abs(a[:n] - spr[:n])))
        n = min(len(b), len(spr)); sR.append(np.max(np.abs(b[:n] - spr[:n])))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    hh, eE0, eR0 = hs[1:], eE[1:], eR[1:]   # con h = 0.4 ya hay un spike antes de t = 3
    ax[0].loglog(hh, eE0, 'o-', color=NARANJA, label='Euler')
    ax[0].loglog(hh, eR0, 'o-', color=AZUL, label='RK4')
    ax[0].loglog(hh, eE0[1]*(hh/hh[1]), ':', color=GRIS, label='∝ h')
    ax[0].loglog(hh, eR0[1]*(hh/hh[1])**4, '--', color=GRIS, label='∝ h⁴')
    ax[0].set_xlabel('h  [ms]'); ax[0].set_ylabel('|v(3) − v_ref(3)|  [mV]')
    ax[0].set_title('Tramo suave: v en t = 3 ms, antes del 1er spike'); ax[0].legend(fontsize=9)
    ax[1].loglog(hs, sE, 'o-', color=NARANJA, label='Euler')
    ax[1].loglog(hs, sR, 'o-', color=AZUL, label='RK4')
    ax[1].loglog(hs, hs, ':', color=GRIS, label='h (un paso)')
    ax[1].set_xlabel('h  [ms]'); ax[1].set_ylabel('máx |t_k − t_k,ref|  [ms]')
    ax[1].set_title('Tiempos de spike en 100 ms'); ax[1].legend(fontsize=9)
    for a, xs in zip(ax, (hh, hs)):
        a.set_xticks(xs); a.set_xticklabels([str(x) for x in xs]); a.minorticks_off()
    fig.suptitle('Lejos del disparo RK4 es de orden 4; el tiempo del spike queda atado a la grilla (error ~ h)', y=1.02)
    guardar(fig, 'r2_orden.png')
    return dict(h=hs.tolist(), eE=eE, eR=eR, sE=sE, sR=sR)


# ------------------------------------------------------------------ R3 el truquito
def r3_truquito():
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1, 1.25]})
    # (a) sin reseteo: v explota en tiempo finito
    t, w, _, _ = correr(rk4, 0.001, T=13.6, con_reset=False)
    ok = np.isfinite(w[0]) & (w[0] < 1e4)
    ax[0].plot(t[ok], w[0, ok], color=ROJO, lw=1.5, label='sin reseteo')
    t2, w2, sp2, _ = correr(rk4, 0.001, T=16)
    ax[0].plot(t2, np.where(np.isin(np.round(t2, 3), np.round(sp2, 3)), 30, w2[0]), color=AZUL, lw=1.2, label='con reseteo')
    ax[0].axhline(30, color=ROJO, ls='--', lw=0.8); ax[0].text(10.2, 45, '$v_+ = 30$', color=ROJO)
    ax[0].set_ylim(-80, 400); ax[0].set_xlim(10, 16)
    ax[0].set_xlabel('t  [ms]'); ax[0].set_ylabel('v  [mV]'); ax[0].legend(fontsize=9, loc='upper left')
    ax[0].set_title('Sin el truquito v → ∞ en tiempo finito')
    ax[0].text(13.65, 250, 'el término 0.04 v²\nmanda: asíntota\nvertical', fontsize=9, color=ROJO)

    # (b) zoom de un spike con la grilla h = 0.1
    h = 0.1
    t, w, sp, prop = correr(rk4, h, T=20)
    j = int(round(sp[0]/h))
    sl = slice(j-14, j+6)
    ax[1].plot(t[sl], w[0, sl], '-', color=AZUL, lw=0.8, alpha=0.6)
    ax[1].plot(t[sl], w[0, sl], 'o', color=AZUL, ms=4.5, label='valores guardados $v_n$')
    ax[1].plot(t[j], prop[0][0], 'o', mfc='white', color=ROJO, ms=8, mew=1.8, label='propuesta de RK4 en $t_{n+1}$')
    ax[1].annotate('', xy=(t[j], w[0, j]), xytext=(t[j], prop[0][0]),
                   arrowprops=dict(arrowstyle='->', color=VERDE, lw=1.6))
    ax[1].text(t[j]+0.06, -15, f'condicionante:\n{prop[0][0]:.1f} ≥ 30  →\nv ← v₋ = −65\nu ← u + Δu', color=VERDE, fontsize=9)
    ax[1].annotate(f'último guardado antes\ndel spike: {w[0, j-1]:.1f} mV', xy=(t[j-1], w[0, j-1]), xytext=(12.15, 0),
                   arrowprops=dict(arrowstyle='->', color=GRIS), fontsize=9, color=GRIS)
    ax[1].axhline(30, color=ROJO, ls='--', lw=0.8); ax[1].axhline(-65, color=VERDE, ls='--', lw=0.8)
    ax[1].set_xlabel('t  [ms]'); ax[1].set_ylabel('v  [mV]'); ax[1].legend(fontsize=8.5, loc='upper left')
    ax[1].set_title(f'Zoom del 1er spike con h = {h} ms'); ax[1].set_ylim(-75, 50)
    guardar(fig, 'r3_truquito.png')


# ------------------------------------------------------------------ R4 caso base
def r4_caso_base():
    t, w, sp, prop = correr(rk4, 0.1)
    v_plot = w[0].copy()
    jj = np.round(sp/0.1).astype(int)
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1], height_ratios=[2.2, 1.3, 1], hspace=0.5, wspace=0.28)
    a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[1, 0], sharex=a0); a2 = fig.add_subplot(gs[2, 0], sharex=a0)
    b0 = fig.add_subplot(gs[0:2, 1]); b1 = fig.add_subplot(gs[2, 1])
    # para dibujar: insertamos el pico en v+ justo en el instante del disparo
    tt = np.insert(t, jj, sp); vv = np.insert(v_plot, jj, 30.0)
    a0.plot(tt, vv, color=AZUL, lw=1, label='con el pico dibujado en 30')
    a0.plot(t, w[0], color=ROJO, lw=0.7, ls='--', label='salida cruda del integrador')
    a0.legend(fontsize=8, loc='lower right', ncol=2, bbox_to_anchor=(1, 1.0), frameon=False)
    a0.set_ylabel('v  [mV]'); a0.set_title('Caso base (TP 1.1)', loc='left', fontsize=10, pad=8)
    fig.suptitle('Caso base (TP 1.1):  b = 0.02,  c = 0.2,  v₋ = −65,  v₊ = 30,  Δu = 2,  I = 10 desde t = 10,  v(0) = −70,  u(0) = c·v(0),  RK4 con h = 0.1', fontsize=10, y=0.995)
    a1.plot(t, w[1], color=NARANJA); a1.set_ylabel('u')
    a2.plot(t, [I_base(s) for s in t], color=GRIS); a2.set_ylabel('I'); a2.set_xlabel('t  [ms]')
    for s in sp:
        a1.axvline(s, color=VIOLETA, lw=0.5, alpha=0.4)
    # plano de fase
    vg = np.linspace(-80, 32, 300)
    b0.plot(vg, 0.04*vg**2 + 5*vg + 140 + 10, color=INK, lw=1.4, label='nulclina v (I=10)')
    b0.plot(vg, 0.2*vg, color=NARANJA, lw=1.4, label='nulclina u')
    cortes = np.where(np.diff(w[0]) < -50)[0] + 1
    for k, seg in enumerate(np.split(np.arange(len(t)), cortes)):
        col = GRIS if k < 4 else AZUL
        b0.plot(w[0, seg], w[1, seg], color=col, lw=1 if k < 4 else 1.6)
    b0.axvline(30, color=ROJO, ls='--', lw=0.8); b0.axvline(-65, color=VERDE, ls='--', lw=0.8)
    b0.set_xlim(-80, 35); b0.set_ylim(-16, 2); b0.set_xlabel('v'); b0.set_ylabel('u')
    b0.set_title('Plano de fase: transitorio (gris)\ny ciclo final (azul)', fontsize=10); b0.legend(fontsize=8, loc='lower right')
    isi = np.diff(sp)
    b1.plot(np.arange(1, len(isi)+1), isi, 'o-', color=VIOLETA)
    b1.set_xlabel('número de intervalo'); b1.set_xticks(range(1, len(isi)+1, 2)); b1.set_ylabel('ISI  [ms]')
    b1.set_title(f'Adaptación: ISI {isi[0]:.1f} → {isi[-1]:.1f} ms (≈{1000/isi[-1]:.0f} Hz)', fontsize=10)
    guardar(fig, 'r4_caso_base.png')
    return dict(spikes=sp.tolist(), isi=isi.tolist(), prop=[float(x[0]) for x in prop],
                guardado_previo=[float(w[0, j-1]) for j in jj])


if __name__ == '__main__':
    r1_paso(); o = r2_orden(); r3_truquito(); b = r4_caso_base()
    print(o); print(b)
