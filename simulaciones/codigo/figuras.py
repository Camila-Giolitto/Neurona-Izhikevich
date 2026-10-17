"""Regenera todas las figuras del TP1 (consigna actualizada, 28/9/2026) en ../figs y ../resultados.json."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json
from izhikevich import *
import red_izhikevich as red

BLUE, ORANGE, AQUA, GRAY, INK, RED, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#e34948', '#4a3aa7'
plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.grid': True, 'grid.color': '#e7e6e2', 'grid.linewidth': 0.6, 'lines.linewidth': 1.2,
    'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight', 'legend.frameon': False})
R = {}
def save(fig, name): fig.savefig(f'../figs/{name}.png'); plt.close(fig)

# ======================= F1: caso base (parte 1.1 y 1.2): t en [0,100], v(0)=-70, u(0)=b v(0)=-14
p = parametros(a=0.02, b=0.2, c=-65, d=2, I=I_base)
t, w = simular(p, -70.0, T=100.0, h=0.1)
js = tiempos_disparo(t, w); ts = t[js]
R['base_spikes'] = ts.round(1).tolist()
R['base_isi'] = np.diff(ts).round(2).tolist()
R['base_vmax_sinpico'] = float(w[0].max())
tt, vv = con_picos(t, w)
fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1.6, 1]})
ax[0].plot(tt, vv, color=BLUE, label='v(t), pico dibujado en 30 mV')
ax[0].plot(ts, np.full(ts.size, 34), 'v', color=RED, ms=5, label='spikes registrados (v ≥ 30 mV)')
ax[0].set_ylabel('v(t) [mV]'); ax[0].set_ylim(-80, 40)
ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
ax[1].plot(t, w[1], color=ORANGE); ax[1].set_ylabel('u(t)')
ax[2].plot(t, [I_base(x) for x in t], color=INK); ax[2].set_ylabel('I(t)'); ax[2].set_xlabel('t [ms]')
ax[2].set_ylim(-1, 12)
for a in ax:
    a.axvline(10, color=GRAY, ls='--', lw=0.9)
    for x in ts: a.axvline(x, color=RED, lw=0.4, alpha=0.35)
ax[2].text(11, 4, 'inicio del estímulo (t = 10 ms)', fontsize=8, color='#52514e')
ax[0].set_xlim(0, 100)
save(fig, 'f1_base')

# ======================= F2: ocho paneles de la figura 2 (protocolos de la consigna)
fig, axs = plt.subplots(4, 2, figsize=(10, 10.5))
R['casos'] = {}
for ax, nombre in zip(axs.flatten(), CASOS):
    t, w = simular_caso(nombre)
    tt, vv = con_picos(t, w)
    I = CASOS[nombre][4]
    Iv = np.array([I(x) for x in t])
    ax.plot(tt, vv, color=BLUE, lw=1)
    lo = -95                                     # corriente dibujada abajo, escalada, como en el paper
    Is = (Iv - Iv.min())/(max(np.ptp(Iv), 1e-9))
    ax.plot(t, lo + 8*Is, color=INK, lw=1)
    ax.set_title(NOMBRES[nombre], fontsize=9.5, loc='left')
    ax.set_ylim(lo - 4, 38); ax.set_xlim(0, t[-1])
    ax.set_yticks([-80, -40, 0, 30])
    jd = tiempos_disparo(t, w)
    R['casos'][nombre] = dict(v0=CASOS[nombre][5], T=DURACION[nombre], n=int(jd.size), t=t[jd].round(1).tolist())
for a in axs[-1]: a.set_xlabel('t [ms]')
for a in axs[:, 0]: a.set_ylabel('v [mV]')
fig.tight_layout(); save(fig, 'f2_figura2')

# ======================= F2b: variable u en los 8 casos
fig, axs = plt.subplots(2, 4, figsize=(11, 4.2))
for ax, nombre in zip(axs.flatten(), CASOS):
    t, w = simular_caso(nombre)
    ax.plot(t, w[1], color=ORANGE); ax.set_title(nombre, fontsize=9, loc='left')
for a in axs[-1]: a.set_xlabel('t [ms]')
axs[0, 0].set_ylabel('u(t)'); axs[1, 0].set_ylabel('u(t)')
fig.tight_layout(); save(fig, 'f2b_u')

# ======================= F3: plano de fase (RS y CH) con nulclinas
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, nombre in zip(axs, ['RS', 'CH']):
    a_, b_, c_, d_, _, v0 = CASOS[nombre]
    t, w = simular_caso(nombre)
    V = np.linspace(-80, 30, 400)
    ax.plot(V, 0.04*V**2 + 5*V + 140, color=GRAY, ls=':', label='nulclina v (I=0)')
    ax.plot(V, 0.04*V**2 + 5*V + 140 + 10, color=INK, ls='--', label='nulclina v (I=10)')
    ax.plot(V, b_*V, color=ORANGE, label='nulclina u: u = b v')
    jr = tiempos_disparo(t, w)
    vv_ = np.insert(w[0], jr, np.nan); uu_ = np.insert(w[1], jr, np.nan); tq = np.insert(t, jr, t[jr])
    m = tq >= 20
    ax.plot(vv_[m], uu_[m], color=BLUE, lw=0.9, label='trayectoria (t>20), saltos de reseteo omitidos')
    ax.axvline(c_, color=VIOLET, lw=1, ls='-.', label=f'reseteo c = {c_} mV')
    ax.axvline(30, color=RED, lw=1, ls='-.', label='pico del algoritmo: 30 mV')
    vr = reposo(b_); ax.plot(vr, b_*vr, 'o', ms=6, color=INK, mfc='white', zorder=5)
    ax.set_xlim(-80, 35); ax.set_ylim(-20, 10 if nombre == 'RS' else 5)
    ax.set_xlabel('v [mV]'); ax.set_ylabel('u'); ax.set_title(NOMBRES[nombre], loc='left', fontsize=9.5)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); save(fig, 'f3_fase')

# ======================= F4: curva f-I + puntos fijos vs I
def bif(a, b):
    """I de silla-nodo y de Hopf del reposo (notación del artículo: a = escala de u, b = sensibilidad)."""
    I_sn = (5 - b)**2/(4*0.04) - 140
    vH = (a - 5)/(2*0.04)                    # traza = 0: 0.08 v + 5 - a = 0
    I_h = -(0.04*vH**2 + (5 - b)*vH + 140)
    det_ok = a*(b - (0.08*vH + 5)) > 0
    return I_sn, (I_h if (det_ok and b > a) else None)
R['bif'] = {k: [round(x, 3) if x is not None else None for x in bif(CASOS[k][0], CASOS[k][1])] for k in ['RS', 'FS', 'LTS', 'RZ']}

Is = np.linspace(0, 20, 81)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
cols = {'RS': BLUE, 'IB': VIOLET, 'CH': AQUA, 'FS': ORANGE, 'LTS': RED}
R['fI'] = {}
for nombre, col in cols.items():
    a_, b_, c_, d_, _, _ = CASOS[nombre]
    fr = []
    for I0 in Is:
        pc = parametros(a=a_, b=b_, c=c_, d=d_, I=lambda t, I0=I0: I0)
        t, w = simular(pc, reposo(b_), T=1000, h=0.1)
        ts = t[tiempos_disparo(t, w)]; ts = ts[ts > 500]
        fr.append(1000*len(ts)/500)
    axs[0].plot(Is, fr, color=col, label=nombre)
    R['fI'][nombre] = dict(I=Is[::8].tolist(), f=np.array(fr)[::8].round(1).tolist())
axs[0].axvspan(3.79, 4.0, color='#e7e6e2'); axs[0].text(4.3, 230, 'pérdida de estabilidad\ndel reposo (I≈3.8–4)', fontsize=8, color='#52514e')
axs[0].set_xlabel('I (constante)'); axs[0].set_ylabel('frecuencia estacionaria [Hz]')
axs[0].legend(ncol=2, fontsize=8); axs[0].set_title('Curva f–I (régimen estacionario, t∈[500,1000] ms)', loc='left', fontsize=9.5)
I_range = np.linspace(-10, 4, 300)
vlo = [reposo(0.2, I) for I in I_range]
vhi = [(-(5 - 0.2) + np.sqrt(4.8**2 - 4*0.04*(140 + I)))/0.08 for I in I_range]
axs[1].plot(I_range, vlo, color=BLUE, label='nodo / foco (reposo)')
axs[1].plot(I_range, vhi, color=BLUE, ls='--', label='silla')
Isn, Ih = bif(0.02, 0.2)
axs[1].plot([Isn], [reposo(0.2, Isn)], 'o', color=INK)
axs[1].annotate(f'silla-nodo\nI={Isn:.2f}', (Isn, reposo(0.2, Isn)), (0.5, -45), fontsize=8, arrowprops=dict(arrowstyle='->', color=GRAY))
if Ih is not None:
    vH = (0.02 - 5)/0.08
    axs[1].plot([Ih], [vH], 's', color=ORANGE)
    axs[1].annotate(f'Hopf\nI={Ih:.2f}', (Ih, vH), (-8, -58), fontsize=8, arrowprops=dict(arrowstyle='->', color=GRAY))
axs[1].set_xlabel('I'); axs[1].set_ylabel('v* [mV]'); axs[1].legend(fontsize=8, loc='lower left')
axs[1].set_title('Puntos fijos vs I (a=0.02, b=0.2)', loc='left', fontsize=9.5)
fig.tight_layout(); save(fig, 'f4_fI')

# ======================= F5: resonancia en RZ (el pulso funciona según la fase de la oscilación)
pRZ = p_caso('RZ'); a_, b_, c_, d_, _, v0 = CASOS['RZ']
I_sin = lambda t: 0.0 if t < 40 else 0.2
t0, w0 = simular(parametros(a_, b_, c_, d_, I_sin), v0, T=300)
t1, w1 = simular_caso('RZ'); tt1, vv1 = con_picos(t1, w1)
tps = np.arange(45, 296, 1.0); exito = []
for tp in tps:
    I = lambda t, tp=tp: 0.0 if t < 40 else (2.2 if tp <= t < tp + 1 else 0.2)
    t, w = simular(parametros(a_, b_, c_, d_, I), v0, T=tp + 50)
    exito.append(len(tiempos_disparo(t, w)) > 0)
exito = np.array(exito)
R['rz_pulso_falla'] = tps[~exito].tolist()
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].plot(tt1, vv1, color=BLUE, lw=1, label='con pulso de 2,2 en [180,181) ms')
axs[0].plot(t0, w0[0], color=INK, lw=1, ls='--', label='sin pulso (I = 0,2 desde 40 ms)')
axs[0].axvspan(180, 181, color=ORANGE, alpha=0.5)
axs[0].set_xlabel('t [ms]'); axs[0].set_ylabel('v [mV]'); axs[0].set_ylim(-75, 35)
axs[0].legend(fontsize=8, loc='upper left'); axs[0].set_title('RZ: oscilación amortiguada y disparo por el pulso', loc='left', fontsize=9.5)
axs[1].plot(t0, w0[0], color=INK, lw=1)
axs[1].plot(tps[exito], np.full(exito.sum(), -58.6), '|', color=AQUA, ms=9, label='el pulso dispara')
axs[1].plot(tps[~exito], np.full((~exito).sum(), -58.6), '|', color=RED, ms=9, label='el pulso no dispara')
axs[1].set_xlim(40, 300); axs[1].set_ylim(-63.5, -58.2)
axs[1].set_xlabel('instante del pulso t_p [ms]'); axs[1].set_ylabel('v sin pulso [mV]')
axs[1].legend(fontsize=8, loc='lower right', ncol=2); axs[1].set_title('¿Dispara un pulso de 1 ms y 2,2 aplicado en t_p?', loc='left', fontsize=9.5)
fig.tight_layout(); save(fig, 'f5_rz')

# ======================= F6: error numérico (Euler vs RK4 vs RK4 + evento), caso RS
from scipy.integrate import solve_ivp
pc = p_caso('RS'); v0RS = CASOS['RS'][5]
def ref_spikes(pc, v0, T=200, cortes=(20.0,)):
    a_, b_, c_, d_, I, vp = pc
    x = np.array([v0, b_*v0]); t0 = 0; out = []
    ev = lambda t, x: x[0] - vp; ev.terminal = True; ev.direction = 1
    while t0 < T:
        tf = min([c for c in cortes if c > t0] + [T])        # integro por tramos (discontinuidad de I)
        s = solve_ivp(lambda t, x: izhikevich(x, t, pc), (t0, tf), x, events=ev, rtol=1e-11, atol=1e-11, method='DOP853', max_step=0.05)
        if s.status == 1:
            ts = s.t_events[0][0]; out.append(ts); xs = s.y_events[0][0]
            x = np.array([c_, xs[1] + d_]); t0 = ts
        else:
            x = s.y[:, -1]; t0 = tf
    return np.array(out)
tref = ref_spikes(pc, v0RS); R['ref_spikes'] = tref.round(4).tolist()
hs = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
err = {'Euler': [], 'RK4': [], 'RK4 + evento': []}
for h in hs:
    k = int(round(200/h))
    for nombre_m, m in [('Euler', euler), ('RK4', rk4)]:
        t, w = simular(pc, v0RS, T=200, h=h, metodo=m)
        ts = t[tiempos_disparo(t, w)]
        err[nombre_m].append(np.max(np.abs(ts - tref)) if len(ts) == len(tref) else np.nan)
    t, w, ts = integrador_eventos(izhikevich, np.array([v0RS, pc[1]*v0RS]), 0, 200, k, pc)
    err['RK4 + evento'].append(np.max(np.abs(ts - tref)) if len(ts) == len(tref) else np.nan)
R['err'] = {k: [float(x) for x in v] for k, v in err.items()}; R['hs'] = hs
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
for (k, v), col in zip(err.items(), [ORANGE, BLUE, AQUA]):
    axs[0].loglog(hs, v, 'o-', color=col, label=k, ms=5)
hh = np.array(hs)
axs[0].loglog(hh, err['RK4'][2]/hh[2]*hh, color=GRAY, ls=':', label='pendiente 1 (∝ h)')
axs[0].loglog(hh, err['RK4 + evento'][3]/hh[3]**2*hh**2, color=GRAY, ls='--', label='pendiente 2 (∝ h²)')
axs[0].set_xticks(hs); axs[0].set_xticklabels([str(x) for x in hs]); axs[0].minorticks_off()
axs[0].set_xlabel('paso h [ms]'); axs[0].set_ylabel('máx |t_k − t_k^ref| [ms]')
axs[0].set_title(f'Error en los tiempos de disparo (RS, {len(tref)} spikes)', loc='left', fontsize=9.5); axs[0].legend(fontsize=8)
for h, col, lab in [(0.5, ORANGE, 'Euler h=0.5'), (0.1, BLUE, 'RK4 h=0.1')]:
    m = euler if 'Euler' in lab else rk4
    t, w = simular(pc, v0RS, T=200, h=h, metodo=m); tt, vv = con_picos(t, w)
    axs[1].plot(tt, vv, color=col, lw=0.9, label=lab)
for x in tref: axs[1].axvline(x, color=GRAY, lw=0.6, ls=':')
axs[1].set_xlim(15, 200); axs[1].set_xlabel('t [ms]'); axs[1].set_ylabel('v [mV]')
axs[1].set_title('RS: Euler vs RK4 (líneas: disparos de referencia)', loc='left', fontsize=9.5); axs[1].legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=2)
fig.tight_layout(); save(fig, 'f6_error')

# ======================= Parte 2: red de 1000 neuronas
T, S0 = 1000, red.SEMILLA
R['semilla'] = S0

def raster(ax, f, ne=800, T=1000):
    e = f[:, 1] < ne
    ax.plot(f[e, 0], f[e, 1], '.', ms=1.2, color=INK, rasterized=True)
    ax.plot(f[~e, 0], f[~e, 1], '.', ms=1.2, color=RED, rasterized=True)
    ax.axhline(ne, color=GRAY, lw=0.8); ax.set_ylim(0, 1000); ax.set_xlim(0, T); ax.grid(False)
    ax.set_ylabel('neurona')

# F7: traducción literal (figura 3)
f_lit = red.red_literal(red.parametros_red(seed=S0), T=T)
R['red_literal'] = red.metricas(f_lit, T); R['red_literal_sin_transitorio'] = red.metricas(f_lit, T, desde=100)
tb, tasa = red.actividad_poblacional(f_lit, T)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
raster(a1, f_lit)
a1.set_title(f'Red de 1000 neuronas, código del artículo traducido literalmente (semilla {S0}). '
             f'{R["red_literal"]["spikes"]} spikes, tasa media {R["red_literal"]["tasa_Hz"]:.1f} Hz', loc='left', fontsize=9.5)
a1.text(15, 400, 'excitatorias', bbox=dict(fc='w', ec='none')); a1.text(15, 900, 'inhibitorias', color=RED, bbox=dict(fc='w', ec='none'))
a2.plot(tb, tasa, color=BLUE, lw=0.7); a2.set_ylabel('tasa pobl. [Hz]'); a2.set_xlabel('t [ms]')
fig.tight_layout(); save(fig, 'f7_red_literal')

# picos de sincronización (bins de 1 ms con muchos spikes) y espectro
f_sp, P = red.espectro(f_lit, T)
m = (f_sp > 2) & (f_sp < 100)
R['red_literal_pico_espectral_Hz'] = float(f_sp[m][np.argmax(P[m])])

# F8: con vs sin autoconexiones (misma semilla)
f_sin = red.red_literal(red.parametros_red(seed=S0, autoconexiones=False), T=T)
R['red_sin_diagonal'] = red.metricas(f_sin, T); R['red_sin_diagonal_sin_transitorio'] = red.metricas(f_sin, T, desde=100)
comp = []
for s in range(1, 21):                                   # robustez: 20 semillas
    m1 = red.metricas(red.red_literal(red.parametros_red(seed=s), T=T), T, desde=100)['tasa_Hz']
    m2 = red.metricas(red.red_literal(red.parametros_red(seed=s, autoconexiones=False), T=T), T, desde=100)['tasa_Hz']
    comp.append((m1, m2))
comp = np.array(comp)
R['diagonal_20_semillas'] = dict(con=[float(comp[:, 0].mean()), float(comp[:, 0].std())],
                                 sin=[float(comp[:, 1].mean()), float(comp[:, 1].std())],
                                 dif=[float((comp[:, 0]-comp[:, 1]).mean()), float((comp[:, 0]-comp[:, 1]).std())])
fig = plt.figure(figsize=(11, 7.2))
gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 1.6], width_ratios=[1, 1])
a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1], sharey=a1)
raster(a1, f_lit); raster(a2, f_sin); a2.set_ylabel('')
a1.set_title(f'(a) literal, con diagonal de S: {R["red_literal"]["spikes"]} spikes, {R["red_literal"]["tasa_Hz"]:.2f} Hz', loc='left', fontsize=9)
a2.set_title(f'(b) sin autoconexiones (s_ii=0): {R["red_sin_diagonal"]["spikes"]} spikes, {R["red_sin_diagonal"]["tasa_Hz"]:.2f} Hz', loc='left', fontsize=9)
a3 = fig.add_subplot(gs[1, :])
tb2, tasa2 = red.actividad_poblacional(f_sin, T)
a3.plot(tb, tasa, color=BLUE, lw=0.7, label='(a) con diagonal'); a3.plot(tb2, tasa2, color=ORANGE, lw=0.7, alpha=0.8, label='(b) sin diagonal')
a3.set_xlim(0, T); a3.set_ylabel('tasa pobl. [Hz]'); a3.set_xlabel('t [ms]'); a3.legend(fontsize=8, ncol=2, loc='upper right')
a4 = fig.add_subplot(gs[2, 0])
f2, P2 = red.espectro(f_sin, T)
a4.plot(f_sp[m], P[m]/P[m].max(), color=BLUE, label='(a)'); a4.plot(f2[m], P2[m]/P[m].max(), color=ORANGE, label='(b)')
a4.set_xlabel('frecuencia [Hz]'); a4.set_ylabel('potencia (norm.)'); a4.legend(fontsize=8)
a4.set_title('Espectro de la tasa poblacional (t > 100 ms)', loc='left', fontsize=9)
a5 = fig.add_subplot(gs[2, 1])
a5.plot(comp[:, 0], comp[:, 1], 'o', color=VIOLET, ms=4)
lo, hi = comp.min() - 0.2, comp.max() + 0.2
a5.plot([lo, hi], [lo, hi], color=GRAY, ls='--', lw=0.8); a5.set_xlim(lo, hi); a5.set_ylim(lo, hi)
a5.set_xlabel('tasa con diagonal [Hz]'); a5.set_ylabel('tasa sin diagonal [Hz]')
a5.set_title('20 semillas (t > 100 ms)', loc='left', fontsize=9)
fig.tight_layout(); save(fig, 'f8_autoconexiones')

# F9: extras opcionales: Euler-Maruyama (h=0.1) y acoplamiento continuo con z(v)
f_em = red.red_euler_maruyama(red.parametros_red(seed=S0), T=T, h=0.1)
f_z = red.red_z(red.parametros_red(seed=S0), T=T)
R['red_em'] = red.metricas(f_em, T); R['red_em_sin_transitorio'] = red.metricas(f_em, T, desde=100)
R['red_z'] = red.metricas(f_z, T); R['red_z_sin_transitorio'] = red.metricas(f_z, T, desde=100)
fig, axs = plt.subplots(2, 2, figsize=(11, 6), gridspec_kw={'height_ratios': [3, 1.3]})
for j, (f_, lab) in enumerate([(f_em, 'Euler–Maruyama, h = 0,1 ms, ruido σ dW'), (f_z, 'acoplamiento continuo I_syn = S z(v)')]):
    raster(axs[0, j], f_)
    key = 'red_em' if j == 0 else 'red_z'
    axs[0, j].set_title(f'{lab}\n{R[key]["spikes"]} spikes; tasa (t>100 ms) {R[key + "_sin_transitorio"]["tasa_Hz"]:.1f} Hz', loc='left', fontsize=9)
    tb_, ta_ = red.actividad_poblacional(f_, T)
    axs[1, j].plot(tb, tasa, color=GRAY, lw=0.6, label='literal (figura 3)')
    axs[1, j].plot(tb_, ta_, color=BLUE if j == 0 else ORANGE, lw=0.7, label='variante')
    axs[1, j].set_xlim(0, T); axs[1, j].set_ylim(0, 120); axs[1, j].set_xlabel('t [ms]'); axs[1, j].legend(fontsize=7.5, ncol=2, loc='upper right')
axs[1, 0].set_ylabel('tasa pobl. [Hz]')
fig.tight_layout(); save(fig, 'f9_extras')

# z(v) vs detector de spikes
fig, ax = plt.subplots(figsize=(5, 3.2))
V = np.linspace(-90, 30, 500)
ax.plot(V, (V >= 30).astype(float), color=INK, label='detector de spikes (v ≥ 30)')
ax.plot(V, red.z(V), color=ORANGE, label='z(v) = (87+v)/450 − 0,0193')
ax.set_xlabel('v [mV]'); ax.legend(fontsize=8); ax.set_title('z(v) no es un detector binario', loc='left', fontsize=9.5)
fig.tight_layout(); save(fig, 'f10_zv')

json.dump(R, open('resultados.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in R.items() if k not in ('fI',)}, indent=1)[:5000])
