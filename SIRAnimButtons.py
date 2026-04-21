import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp

from ModelPresets import PRESETS, PRESET_ICS, DEFAULT_ICS

# ── Initial parameters ────────────────────────────────────────────────────────
NUM_SUSCEPTIBLE = 999
TOTAL_AGENTS    = NUM_SUSCEPTIBLE  # S only at start; I and P come from IC sliders

DEFAULT_PARAMS = dict(
    beta=5.0, gamma=0.5, epsilon=0.1,
    phi=0.02,  mu=0.3,   alpha=0.05,
    delta=0.1, c=1.0
)

T_MAX    = 100
ARENA_R  = 1.0
N_FRAMES = 600

COL = dict(S="#4fc3f7", I="#ef5350", R="#66bb6a", D="#78909c", P="#ffa726")
COLORS_RGBA = {
    'S': [0.31, 0.76, 0.97, 0.85],
    'I': [0.94, 0.33, 0.31, 0.85],
    'R': [0.40, 0.73, 0.41, 0.85],
    'D': [0.47, 0.56, 0.61, 0.85],
    'P': [1.00, 0.65, 0.15, 0.85],
}

BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
ACCENT  = "#58a6ff"

# ── Static dot positions ──────────────────────────────────────────────────────
rng      = np.random.default_rng(42)
N_DOTS   = NUM_SUSCEPTIBLE + 200   # headroom for births/predators
angles_d = rng.uniform(0, 2*np.pi, N_DOTS)
radii_d  = np.sqrt(rng.uniform(0, 1, N_DOTS)) * ARENA_R * 0.92
dot_x    = radii_d * np.cos(angles_d)
dot_y    = radii_d * np.sin(angles_d)

# ── ODE solver ────────────────────────────────────────────────────────────────
def run_odes(p, i0, p0):
    s0 = max(NUM_SUSCEPTIBLE - int(i0), 0)
    def odes(t, y):
        S, I, R, P, D = y
        S, I, R, P = max(S,0), max(I,0), max(R,0), max(P,0)
        N = max(S+I+R, 1e-6)
        dS  = p['mu']*N - p['beta']*S*I/N - p['phi']*P*S
        dI  = p['beta']*S*I/N - p['gamma']*I - p['epsilon']*I - p['phi']*p['c']*P*I
        dR  = p['gamma']*I - p['phi']*P*R
        dPr = p['alpha']*p['phi']*P*N - p['delta']*P
        dD  = p['epsilon']*I + p['phi']*P*N
        return [dS, dI, dR, dPr, dD]

    sol = solve_ivp(
        odes, [0, T_MAX],
        [s0, i0, 0, p0, 0],
        method='RK45', max_step=0.1, dense_output=True,
        t_eval=np.linspace(0, T_MAX, N_FRAMES),
    )
    return sol.t, sol.y

T_vals, sol_y = run_odes(DEFAULT_PARAMS, DEFAULT_ICS['i0'], DEFAULT_ICS['p0'])
S_vals, I_vals, R_vals, P_vals, D_vals = sol_y

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9), facecolor=BG)

gs_top = gridspec.GridSpec(
    1, 2, width_ratios=[1, 1.5],
    left=0.03, right=0.97, top=0.93, bottom=0.44,
    wspace=0.06
)
ax_sim  = fig.add_subplot(gs_top[0])
ax_line = fig.add_subplot(gs_top[1])

# ── Rate sliders (2-column, 4 rows) ──────────────────────────────────────────
SLIDER_PARAMS = [
    ('beta',    r'β  Transmission',   0.1, 10.0),
    ('gamma',   r'γ  Recovery',       0.0,  2.0),
    ('epsilon', r'ε  Mortality',      0.0,  1.0),
    ('phi',     r'φ  Predation rate', 0.0,  1.0),
    ('mu',      r'μ  Birth rate',     0.0,  0.3),
    ('alpha',   r'α  Pred. effic.',   0.0,  0.2),
    ('delta',   r'δ  Pred. death',    0.0,  0.5),
    ('c',       r'c  Selectivity',    0.5,  8.0),
]

slider_objs = {}
COL_W  = 0.38
SL_H   = 0.022
SL_GAP = 0.036
SL_TOP = 0.40

for i, (key, label, vmin, vmax) in enumerate(SLIDER_PARAMS):
    col  = i % 2
    row  = i // 2
    left = 0.06 + col * (COL_W + 0.08)
    bot  = SL_TOP - row * SL_GAP
    ax_sl = fig.add_axes([left, bot, COL_W * 0.75, SL_H], facecolor=PANEL)
    sl    = Slider(ax_sl, '', vmin, vmax, valinit=DEFAULT_PARAMS[key],
                   color=ACCENT, track_color=BORDER)
    sl.label.set_visible(False)
    sl.valtext.set_color(SUBTEXT)
    sl.valtext.set_fontsize(7)
    ax_sl.set_title(label, color=SUBTEXT, fontsize=7.5,
                    loc='left', pad=2, fontfamily='monospace')
    slider_objs[key] = sl

# ── IC sliders (full-width, below rate sliders) ───────────────────────────────
IC_SLIDERS = [
    ('i0', 'I₀  Initial infected',  1, 100),
    ('p0', 'P₀  Initial predators', 0, 100),
]
IC_TOP  = SL_TOP - 4 * SL_GAP - 0.01
IC_W    = 0.38

for i, (key, label, vmin, vmax) in enumerate(IC_SLIDERS):
    col  = i % 2
    left = 0.06 + col * (IC_W + 0.08)
    ax_sl = fig.add_axes([left, IC_TOP, IC_W * 0.75, SL_H], facecolor=PANEL)
    sl    = Slider(ax_sl, '', vmin, vmax, valinit=DEFAULT_ICS[key],
                   valstep=1, color="#c084fc", track_color=BORDER)
    sl.label.set_visible(False)
    sl.valtext.set_color(SUBTEXT)
    sl.valtext.set_fontsize(7)
    ax_sl.set_title(label, color="#c084fc", fontsize=7.5,
                    loc='left', pad=2, fontfamily='monospace')
    slider_objs[key] = sl

# ── Preset buttons ────────────────────────────────────────────────────────────
PRESET_NAMES = list(PRESETS.keys())
N_PRE   = len(PRESET_NAMES)
PRE_W   = 0.088
PRE_GAP = 0.010
PRE_BOT = 0.022
PRE_H   = 0.065

preset_btns = []
for j, name in enumerate(PRESET_NAMES):
    ax_b = fig.add_axes([0.03 + j*(PRE_W+PRE_GAP), PRE_BOT, PRE_W, PRE_H],
                        facecolor=PANEL)
    btn  = Button(ax_b, name, color=PANEL, hovercolor="#21262d")
    btn.label.set_color(SUBTEXT)
    btn.label.set_fontsize(6.5)
    btn.label.set_fontfamily('monospace')
    preset_btns.append(btn)

# ── Reset button ──────────────────────────────────────────────────────────────
reset_left = 0.03 + N_PRE * (PRE_W + PRE_GAP) + 0.012
ax_reset   = fig.add_axes([reset_left, PRE_BOT, 0.07, PRE_H], facecolor=PANEL)
btn_reset  = Button(ax_reset, "↺  Reset", color=PANEL, hovercolor="#21262d")
btn_reset.label.set_color(ACCENT)
btn_reset.label.set_fontsize(8)
btn_reset.label.set_fontfamily('monospace')

# ── Sim panel ─────────────────────────────────────────────────────────────────
ax_sim.set_facecolor(BG)
ax_sim.set_aspect('equal')
ax_sim.set_xlim(-1.12, 1.12)
ax_sim.set_ylim(-1.12, 1.12)
ax_sim.axis('off')
theta_c = np.linspace(0, 2*np.pi, 300)
ax_sim.plot(np.cos(theta_c), np.sin(theta_c), color=BORDER, lw=1.5, zorder=0)

scat     = ax_sim.scatter(dot_x, dot_y, s=3, c=COL['S'],
                          linewidths=0, alpha=0.85, zorder=2)
time_txt = ax_sim.text(0, -1.07, "t = 0.00", ha='center', va='top',
                       color=SUBTEXT, fontsize=8, fontfamily='monospace')

# ── Chart panel ───────────────────────────────────────────────────────────────
ax_line.set_facecolor(BG)
for sp in ax_line.spines.values():
    sp.set_color(BORDER)
ax_line.tick_params(colors=SUBTEXT, labelsize=8)
ax_line.set_xlabel("Time",       color=SUBTEXT, fontsize=9)
ax_line.set_ylabel("Population", color=SUBTEXT, fontsize=9)
ax_line.set_xlim(0, T_MAX)

trace_lines = {}
for k, arr in [('S',S_vals),('I',I_vals),('R',R_vals),('D',D_vals),('P',P_vals)]:
    ln, = ax_line.plot(T_vals, arr, color=COL[k], lw=1.2, alpha=0.4)
    trace_lines[k] = ln

infected_fill = [ax_line.fill_between(T_vals, I_vals, alpha=0.07, color=COL['I'])]

live_dots = {
    k: ax_line.plot([], [], 'o', color=COL[k], ms=5, zorder=5)[0]
    for k in ('S','I','R','D','P')
}
time_line = ax_line.axvline(0, color=ACCENT, lw=0.8, alpha=0.6, zorder=4)

legend_handles = [
    Line2D([0],[0], marker='o', color='none',
           markerfacecolor=COL[k], markersize=7, label=lbl)
    for k, lbl in [('S','Susceptible'),('I','Infected'),('R','Recovered'),
                   ('D','Dead'),('P','Predator')]
]
ax_line.legend(handles=legend_handles, loc='upper right',
               facecolor=PANEL, edgecolor=BORDER,
               labelcolor="#c9d1d9", fontsize=8)
ax_line.set_ylim(0, max(S_vals.max(), 10) * 1.05)

fig.text(0.5, 0.97, "SIR + Predator  Model", ha='center', va='top',
         color=TEXT, fontsize=14, fontweight='bold', fontfamily='monospace')

# ── State ─────────────────────────────────────────────────────────────────────
state = {'frame': 0}
_updating = [False]

# ── Redraw ────────────────────────────────────────────────────────────────────
def redraw_traces(_=None):
    if _updating[0]:
        return
    global T_vals, S_vals, I_vals, R_vals, P_vals, D_vals

    p  = {k: slider_objs[k].val for k in DEFAULT_PARAMS}
    i0 = int(slider_objs['i0'].val)
    p0 = int(slider_objs['p0'].val)

    T_vals, sol_y2 = run_odes(p, i0, p0)
    S_vals, I_vals, R_vals, P_vals, D_vals = sol_y2

    for k in list(trace_lines.keys()):
        trace_lines[k].remove()
    for k, arr in [('S',S_vals),('I',I_vals),('R',R_vals),('D',D_vals),('P',P_vals)]:
        ln, = ax_line.plot(T_vals, arr, color=COL[k], lw=1.2, alpha=0.4)
        trace_lines[k] = ln

    infected_fill[0].remove()
    infected_fill[0] = ax_line.fill_between(T_vals, I_vals, alpha=0.07, color=COL['I'])

    ymax = max(S_vals.max(), I_vals.max(), R_vals.max(),
               D_vals.max(), P_vals.max(), 10) * 1.05
    ax_line.set_ylim(0, ymax)
    state['frame'] = 0
    ani._init_drawn = False
    fig.canvas.draw()

for sl in slider_objs.values():
    sl.on_changed(redraw_traces)

# ── Preset & reset callbacks ──────────────────────────────────────────────────
def make_preset_cb(name, preset_dict):
    def cb(event):
        _updating[0] = True
        for k, v in preset_dict.items():
            slider_objs[k].set_val(v)
        # Apply IC overrides for this preset if any
        ic_overrides = PRESET_ICS.get(name, {})
        for k, v in ic_overrides.items():
            slider_objs[k].set_val(v)
        _updating[0] = False
        redraw_traces()
    return cb

for btn, name in zip(preset_btns, PRESET_NAMES):
    btn.on_clicked(make_preset_cb(name, PRESETS[name]))

def on_reset(event):
    _updating[0] = True
    for k, v in DEFAULT_PARAMS.items():
        slider_objs[k].set_val(v)
    for k, v in DEFAULT_ICS.items():
        slider_objs[k].set_val(v)
    _updating[0] = False
    redraw_traces()

btn_reset.on_clicked(on_reset)

# ── Animation ─────────────────────────────────────────────────────────────────
def update(frame):
    f = state['frame'] % len(T_vals)
    state['frame'] += 1

    s = max(int(round(S_vals[f])), 0)
    i = max(int(round(I_vals[f])), 0)
    r = max(int(round(R_vals[f])), 0)
    d = max(int(round(D_vals[f])), 0)
    p = max(int(round(P_vals[f])), 0)

    colour_array = np.zeros((N_DOTS, 4))
    idx = 0
    for k, count in [('S',s),('I',i),('R',r),('D',d),('P',p)]:
        end = min(idx + count, N_DOTS)
        colour_array[idx:end] = COLORS_RGBA[k]
        idx = end
    scat.set_facecolor(colour_array)

    t = T_vals[f]
    time_txt.set_text(f"t = {t:.1f}")
    time_line.set_xdata([t, t])
    for k, arr in [('S',S_vals),('I',I_vals),('R',R_vals),('D',D_vals),('P',P_vals)]:
        live_dots[k].set_data([t], [arr[f]])

    return scat, time_txt, time_line, *live_dots.values()

ani = FuncAnimation(fig, update, interval=0, blit=False, cache_frame_data=False)

plt.show()