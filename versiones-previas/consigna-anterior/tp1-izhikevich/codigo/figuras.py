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

# ======================= F1: caso base (parte 1.1 y 1.2)
p = parametros(b=0.02, c=0.2, v_menos=-65, du=2, I=I_base)
t, w = integrador_ode(rk4, izhikevich, np.array([-70, 0.2*-70]), 0, 200, 2000, p, reseteo)
js = tiempos_disparo(t, w, -65); ts = t[js]
R['base_spikes'] = ts.round(1).tolist()
R['base_isi'] = np.diff(ts).round(2).tolist()
R['base_vmax_sinpico'] = float(w[0].max())
tt, vv = con_picos(t, w)
fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1.6, 1]})
ax[0].plot(tt, vv, color=BLUE, label='con picos igualados a $v^+$=30 mV')
ax[0].plot(t, w[0], color=INK, lw=0.7, ls='--', label='salida cruda del integrador (máx ≈ %.0f mV)' % w[0].max(), zorder=3)
ax[0].set_ylabel('v(t) [mV]'); ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
ax[1].plot(t, w[1], color=ORANGE); ax[1].set_ylabel('u(t)')
ax[2].plot(t, [I_base(x) for x in t], color=INK); ax[2].set_ylabel('I(t)'); ax[2].set_xlabel('t [ms]')
ax[2].set_ylim(-1, 12)
save(fig, 'f1_base')

# ======================= F2: ocho paneles de la figura 2
fig, axs = plt.subplots(4, 2, figsize=(10, 10.5))
R['casos'] = {}
for ax, nombre in zip(axs.T.flatten() if False else axs.flatten(), CASOS):
    pc = p_caso(nombre); v0 = reposo(pc[6])
    t, w = simular(pc, v0=v0)
    tt, vv = con_picos(t, w)
    Iv = np.array([pc[8](x) for x in t])
    ax.plot(tt, vv, color=BLUE, lw=1)
    # corriente dibujada abajo, escalada, como en el paper
    lo = min(vv.min(), -90)
    Is = (Iv - Iv.min())/(max(np.ptp(Iv), 1e-9))
    ax.plot(t, lo - 12 + 8*Is, color=INK, lw=1)
    ax.set_title(f'{NOMBRES[nombre]}', fontsize=9.5, loc='left')
    ax.set_ylim(lo - 16, 38); ax.set_xlim(0, 200)
    ax.set_yticks([-80, -40, 0, 30])
    nsp = len(tiempos_disparo(t, w, None))
    R['casos'][nombre] = dict(v0=round(float(v0), 2), n=int(nsp), t=t[tiempos_disparo(t, w, None)].round(1).tolist())
for a in axs[-1]: a.set_xlabel('t [ms]')
for a in axs[:, 0]: a.set_ylabel('v [mV]')
fig.tight_layout(); save(fig, 'f2_figura2')

# ======================= F2b: variable u en los 8 casos (útil para la discusión)
fig, axs = plt.subplots(2, 4, figsize=(11, 4.2), sharex=True)
for ax, nombre in zip(axs.flatten(), CASOS):
    pc = p_caso(nombre); t, w = simular(pc)
    ax.plot(t, w[1], color=ORANGE); ax.set_title(nombre, fontsize=9, loc='left')
for a in axs[-1]: a.set_xlabel('t [ms]')
axs[0, 0].set_ylabel('u(t)'); axs[1, 0].set_ylabel('u(t)')
fig.tight_layout(); save(fig, 'f2b_u')

# ======================= F3: plano de fase (RS e IB) con nulclinas
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, nombre in zip(axs, ['RS', 'CH']):
    pc = p_caso(nombre); b, c, vm, du = pc[5], pc[6], pc[3], pc[7]
    t, w = simular(pc)
    m = t >= 25
    V = np.linspace(-80, 30, 400)
    ax.plot(V, 0.04*V**2 + 5*V + 140 + 0, color=GRAY, ls=':', label='nulclina v (I=0)')
    ax.plot(V, 0.04*V**2 + 5*V + 140 + 10, color=INK, ls='--', label='nulclina v (I=10)')
    ax.plot(V, c*V, color=ORANGE, label='nulclina u: u = c v')
    vv_, uu_ = w[0].copy(), w[1].copy(); jr = tiempos_disparo(t, w, None)
    vv_ = np.insert(vv_, jr, np.nan); uu_ = np.insert(uu_, jr, np.nan); tq = np.insert(t, jr, t[jr])
    m = tq >= 25
    ax.plot(vv_[m], uu_[m], color=BLUE, lw=0.9, label='trayectoria (t>25), saltos de reset omitidos')
    ax.axvline(vm, color=VIOLET, lw=1, ls='-.', label=f'reset $v^-$={vm}')
    ax.axvline(30, color=RED, lw=1, ls='-.', label='umbral $v^+$=30')
    v0 = reposo(c); ax.plot(v0, c*v0, 'o', ms=6, color=INK, mfc='white', zorder=5)
    ax.set_xlim(-80, 35); ax.set_ylim(-20, 10 if nombre=='RS' else 5)
    ax.set_xlabel('v [mV]'); ax.set_ylabel('u'); ax.set_title(NOMBRES[nombre], loc='left', fontsize=9.5)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); save(fig, 'f3_fase')

# ======================= F4: análisis lineal + curva f-I
def bif(b, c, g2=0.04, g1=5, g0=140):
    """I de saddle-node y de Hopf del punto de reposo (notación del TP: b=escala, c=sensibilidad)."""
    I_sn = (g1 - c)**2/(4*g2) - g0
    vH = (b - g1)/(2*g2)          # traza = 0: 2 g2 v + g1 - b = 0
    I_h = -(g2*vH**2 + (g1 - c)*vH + g0)
    det_ok = b*(c - (2*g2*vH + g1)) > 0
    return I_sn, (I_h if (det_ok and c > b) else None)
R['bif'] = {k: [round(x, 3) if x is not None else None for x in bif(CASOS[k][0], CASOS[k][1])] for k in ['RS', 'FS', 'LTS', 'RZ']}

Is = np.linspace(0, 20, 81)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
cols = {'RS': BLUE, 'IB': VIOLET, 'CH': AQUA, 'FS': ORANGE, 'LTS': RED}
R['fI'] = {}
for nombre, col in cols.items():
    b, c, vm, du, _ = CASOS[nombre]
    fr = []
    for I0 in Is:
        pc = parametros(b=b, c=c, v_menos=vm, du=du, I=lambda t, I0=I0: I0)
        t, w = simular(pc, T=1000, h=0.1)
        ts = t[tiempos_disparo(t, w, None)]
        ts = ts[ts > 500]
        fr.append(1000*len(ts)/500)
    axs[0].plot(Is, fr, color=col, label=nombre)
    R['fI'][nombre] = dict(I=Is[::8].tolist(), f=np.array(fr)[::8].round(1).tolist())
axs[0].axvspan(3.79, 4.0, color='#e7e6e2'); axs[0].text(4.3, 230, 'pérdida de estabilidad\ndel reposo (I≈3.8–4)', fontsize=8, color='#52514e')
axs[0].set_xlabel('I (constante)'); axs[0].set_ylabel('frecuencia estacionaria [Hz]')
axs[0].legend(ncol=2, fontsize=8); axs[0].set_title('Curva f–I (régimen estacionario, t∈[500,1000] ms)', loc='left', fontsize=9.5)
# diagrama del punto fijo para c=0.2
I_range = np.linspace(-10, 4, 300)
vlo = [reposo(0.2, I) for I in I_range]
vhi = [(-(5 - 0.2) + np.sqrt((4.8)**2 - 4*0.04*(140 + I)))/(0.08) for I in I_range]
axs[1].plot(I_range, vlo, color=BLUE, label='nodo / foco (reposo)')
axs[1].plot(I_range, vhi, color=BLUE, ls='--', label='silla (umbral)')
Isn, Ih = bif(0.02, 0.2)
axs[1].plot([Isn], [reposo(0.2, Isn)], 'o', color=INK)
axs[1].annotate(f'saddle-node\nI={Isn:.2f}', (Isn, reposo(0.2, Isn)), (0.5, -45), fontsize=8, arrowprops=dict(arrowstyle='->', color=GRAY))
if Ih is not None:
    vH = (0.02 - 5)/0.08
    axs[1].plot([Ih], [vH], 's', color=ORANGE)
    axs[1].annotate(f'Hopf\nI={Ih:.2f}', (Ih, vH), (-8, -58), fontsize=8, arrowprops=dict(arrowstyle='->', color=GRAY))
axs[1].set_xlabel('I'); axs[1].set_ylabel('v* [mV]'); axs[1].legend(fontsize=8, loc='lower left')
axs[1].set_title('Puntos fijos vs I (b=0.02, c=0.2)', loc='left', fontsize=9.5)
fig.tight_layout(); save(fig, 'f4_fI')

# ======================= F5: sensibilidad a la condición inicial
fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
for ax, nombre in zip(axs, ['TC1', 'RZ', 'LTS']):
    pc = p_caso(nombre); c = pc[6]
    for v0, col, lab in [(-70, ORANGE, 'v(0)=−70'), (reposo(c), BLUE, f'v(0)=v* = {reposo(c):.1f}')]:
        t, w = simular(pc, v0=v0); tt, vv = con_picos(t, w)
        ax.plot(tt, vv, color=col, lw=0.9, label=lab, alpha=0.9 if col == BLUE else 0.8)
    ax.set_title(NOMBRES[nombre], fontsize=9, loc='left'); ax.set_xlabel('t [ms]')
axs[0].set_ylabel('v [mV]')
for ax in axs: ax.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, -0.42), ncol=2)
fig.tight_layout(); save(fig, 'f5_ci')

# ======================= F6: error numérico (Euler vs RK4 vs RK4 + evento)
from scipy.integrate import solve_ivp
pc = p_caso('RS')
def ref_spikes(pc, T=200):
    g2, g1, g0, vm, vp, b, c, du, I = pc
    x = np.array([reposo(c), c*reposo(c)]); t0 = 0; out = []
    ev = lambda t, x: x[0] - vp; ev.terminal = True; ev.direction = 1
    brk = [25.0]
    while t0 < T:
        # integro por tramos para respetar la discontinuidad de I en t=25
        tf = min([b_ for b_ in brk if b_ > t0] + [T])
        s = solve_ivp(lambda t, x: izhikevich(x, t, pc), (t0, tf), x, events=ev, rtol=1e-11, atol=1e-11, method='DOP853', max_step=0.05)
        if s.status == 1:
            ts = s.t_events[0][0]; out.append(ts); xs = s.y_events[0][0]
            x = np.array([vm, xs[1] + du]); t0 = ts
        else:
            x = s.y[:, -1]; t0 = tf
    return np.array(out)
tref = ref_spikes(pc); R['ref_spikes'] = tref.round(4).tolist()
hs = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
err = {'Euler': [], 'RK4': [], 'RK4 + evento': []}
for h in hs:
    k = int(round(200/h))
    for nombre_m, m in [('Euler', euler), ('RK4', rk4)]:
        t, w = simular(pc, h=h, metodo=m)
        ts = t[tiempos_disparo(t, w, None)]
        n = min(len(ts), len(tref)); e = np.max(np.abs(ts[:n] - tref[:n])) if len(ts) == len(tref) else np.nan
        err[nombre_m].append(e)
    v0 = reposo(pc[6]); t, w, ts = integrador_eventos(izhikevich, np.array([v0, pc[6]*v0]), 0, 200, k, pc)
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
axs[0].set_title('Error en los tiempos de disparo (RS, 5 spikes)', loc='left', fontsize=9.5); axs[0].legend(fontsize=8)
for h, col, lab in [(0.5, ORANGE, 'Euler h=0.5'), (0.1, BLUE, 'RK4 h=0.1')]:
    m = euler if 'Euler' in lab else rk4
    t, w = simular(pc, h=h, metodo=m); tt, vv = con_picos(t, w)
    axs[1].plot(tt, vv, color=col, lw=0.9, label=lab)
for x in tref: axs[1].axvline(x, color=GRAY, lw=0.6, ls=':')
axs[1].set_xlim(20, 200); axs[1].set_xlabel('t [ms]'); axs[1].set_ylabel('v [mV]')
axs[1].set_title('RS: Euler vs RK4 (líneas: disparos de referencia)', loc='left', fontsize=9.5); axs[1].legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=2)
fig.tight_layout(); save(fig, 'f6_error')

# ======================= Parte 2: red
T = 1000.0
fir = red.red_matlab_literal(seed=3)
p_red = red.parametros_red(seed=1)
ts_t, is_t, tr, v1_t = red.simular_red(p_red, T=T, tau=0.5, activacion=red.theta, seed=2)
ts_z, is_z, tr_z, v1_z = red.simular_red(p_red, T=T, tau=0.5, activacion=red.z, seed=2)
R['rate_matlab'] = len(fir)/(1000*1.0); R['rate_theta'] = ts_t.size/(1000*T*1e-3); R['rate_z'] = ts_z.size/(1000*T*1e-3)

def raster(ax, ts, idx, ne=800):
    e = idx < ne
    ax.plot(ts[e], idx[e], '.', ms=1.2, color=INK, rasterized=True)
    ax.plot(ts[~e], idx[~e], '.', ms=1.2, color=RED, rasterized=True)
    ax.axhline(ne, color=GRAY, lw=0.8); ax.set_ylim(0, 1000); ax.set_xlim(0, 1000); ax.grid(False)
    ax.set_ylabel('neurona')

fig, ax = plt.subplots(figsize=(10, 3.4))
raster(ax, fir[:, 0].astype(float), fir[:, 1]); ax.set_xlabel('t [ms]')
ax.set_title(f'Traducción literal del MATLAB (paso 1 ms = 2×0.5 ms). Tasa media {R["rate_matlab"]:.1f} Hz', loc='left', fontsize=9.5)
fig.tight_layout(); save(fig, 'f7_matlab')

for nombre, (ts_, is_, tr_, v1_) in {'theta': (ts_t, is_t, tr, v1_t), 'z': (ts_z, is_z, tr_z, v1_z)}.items():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    raster(a1, ts_, is_)
    lab = 'Θ (Heaviside)' if nombre == 'theta' else 'z(v)'
    rate = ts_.size/(1000*T*1e-3)
    a1.set_title(f'Euler–Maruyama, τ=0.5 ms, activación {lab}. Tasa media {rate:.1f} Hz', loc='left', fontsize=9.5)
    a1.text(15, 400, 'excitatorias', bbox=dict(fc='w', ec='none')); a1.text(15, 900, 'inhibitorias', color=RED, bbox=dict(fc='w', ec='none'))
    a2.plot(tr_, v1_, color=BLUE, lw=0.7); a2.set_ylabel('v₁(t) [mV]'); a2.set_xlabel('t [ms]')
    fig.tight_layout(); save(fig, f'f8_red_{nombre}')

# activaciones y espectros
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5))
V = np.linspace(-90, 30, 500)
axs[0].plot(V, (V >= 30).astype(float), color=INK, label='Θ(v − v⁺)')
axs[0].plot(V, red.z(V, p_red), color=ORANGE, label='z(v) = (87+v)/450 − 0.0193')
axs[0].set_xlabel('v [mV]'); axs[0].set_ylabel('activación'); axs[0].legend(fontsize=8)
axs[0].set_title('Funciones de activación', loc='left', fontsize=9.5)
R['spec'] = {}
from scipy.signal import welch
TL = 5000.0
for (act, nombre, col) in [(red.theta, 'Θ', BLUE), (red.z, 'z(v)', ORANGE)]:
    ts_, *_ = red.simular_red(p_red, T=TL, tau=0.5, activacion=act, seed=5)
    tb, tasa, frec, pot = red.actividad_poblacional(ts_, TL, 1000, bin_ms=1.0)
    m_t = (tb >= 200) & (tb < 1200)
    axs[1].plot(tb[m_t], tasa[m_t], color=col, lw=0.7, label=nombre, alpha=0.85)
    x = tasa[tb >= 200]
    fw, Pw = welch(x - x.mean(), fs=1000, nperseg=512)
    m = (fw > 2) & (fw < 100)
    axs[2].plot(fw[m], Pw[m]/Pw[m].max(), color=col, lw=1.2, label=nombre)
    R['spec'][nombre] = dict(pico=float(fw[m][np.argmax(Pw[m])]), rate=float(x.mean()), cv=float(x.std()/x.mean()))
axs[1].set_xlabel('t [ms]'); axs[1].set_ylabel('tasa poblacional [Hz/neurona]'); axs[1].legend(fontsize=8)
axs[1].set_title('Actividad poblacional (1 s de una corrida de 5 s)', loc='left', fontsize=9.5)
for f0 in (10, 40): axs[2].axvline(f0, ls='--', color=GRAY, lw=0.8)
axs[2].set_xlabel('frecuencia [Hz]'); axs[2].set_ylabel('potencia (normalizada)'); axs[2].legend(fontsize=8)
axs[2].set_title('Espectro de Welch (5 s; líneas: 10 Hz α, 40 Hz γ)', loc='left', fontsize=9.5)
fig.tight_layout(); save(fig, 'f9_espectro')

json.dump(R, open('../resultados.json', 'w'), indent=1)
print(json.dumps(R, indent=1)[:6000])
