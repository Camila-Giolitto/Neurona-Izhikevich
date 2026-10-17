"""Figura compacta de la segunda parte: red literal, sin diagonal, Euler-Maruyama y z(v). Mismo código que figuras.py."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
import red_izhikevich as red
fm.fontManager.addfont('/usr/share/fonts/opentype/inter/Inter-Regular.otf')
BLUE, ORANGE, AQUA, GRAY, INK, RED, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#e34948', '#4a3aa7'
plt.rcParams.update({'font.family': 'Inter', 'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.grid': True, 'grid.color': '#e7e6e2', 'grid.linewidth': 0.5, 'legend.frameon': False})
T, S0 = 1000, red.SEMILLA
runs = {'lit': red.red_literal(red.parametros_red(seed=S0), T=T),
        'sin': red.red_literal(red.parametros_red(seed=S0, autoconexiones=False), T=T),
        'em':  red.red_euler_maruyama(red.parametros_red(seed=S0), T=T, h=0.1),
        'z':   red.red_z(red.parametros_red(seed=S0), T=T)}
for k, f in runs.items():
    print(k, red.metricas(f, T), red.metricas(f, T, desde=100))
tit = {'lit': '(a) literal (con diagonal de S)', 'sin': '(b) sin autoconexiones, $s_{ii}=0$',
       'em': '(c) Euler–Maruyama, $h=0{,}1$ ms', 'z': '(d) acoplamiento con $z(v)$'}
fig = plt.figure(figsize=(7.2, 3.6))
gs = fig.add_gridspec(2, 4, height_ratios=[2.3, 1])
for j, k in enumerate(runs):
    ax = fig.add_subplot(gs[0, j]); f = runs[k]; e = f[:, 1] < 800
    ax.plot(f[e, 0], f[e, 1], '.', ms=0.6, mew=0, color=INK, rasterized=True)
    ax.plot(f[~e, 0], f[~e, 1], '.', ms=0.6, mew=0, color=RED, rasterized=True)
    ax.axhline(800, color=GRAY, lw=0.6); ax.set_xlim(0, T); ax.set_ylim(0, 1000); ax.grid(False)
    ax.set_title(tit[k], loc='left', fontsize=7.5); ax.set_xlabel('t [ms]')
    if j == 0: ax.set_ylabel('neurona')
    else: ax.tick_params(labelleft=False)
tb, ta = {}, {}
for k in runs: tb[k], ta[k] = red.actividad_poblacional(runs[k], T)
a1 = fig.add_subplot(gs[1, :2]); a2 = fig.add_subplot(gs[1, 2:], sharey=a1)
a1.plot(tb['lit'], ta['lit'], color=BLUE, lw=0.6, label='(a)'); a1.plot(tb['sin'], ta['sin'], color=ORANGE, lw=0.6, alpha=0.85, label='(b)')
a2.plot(tb['lit'], ta['lit'], color=GRAY, lw=0.5, label='(a)'); a2.plot(tb['em'], ta['em'], color=AQUA, lw=0.6, label='(c)'); a2.plot(tb['z'], ta['z'], color=VIOLET, lw=0.6, alpha=0.85, label='(d)')
for a in (a1, a2):
    a.set_xlim(0, T); a.set_ylim(0, 120); a.set_xlabel('t [ms]'); a.legend(ncol=3, loc='upper right', fontsize=7)
a1.set_ylabel('tasa pobl. [Hz]'); a2.tick_params(labelleft=False)
fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.6)
fig.savefig(sys.argv[1], dpi=300)
