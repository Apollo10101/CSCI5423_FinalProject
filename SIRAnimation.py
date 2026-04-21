import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

# ── Parameters ────────────────────────────────────────────────────────────────
NUM_SUSCEPTIBLE = 999
NUM_INFECTED    = 1
NUM_RECOVERED   = 0
NUM_PREDATOR    = 10
TOTAL_AGENTS    = NUM_SUSCEPTIBLE + NUM_INFECTED + NUM_RECOVERED

BETA    = 2.0
GAMMA   = 0.5
EPSILON = 0.1
PHI     = 0.1
MU      = 0.05
ALPHA   = 0.05
DELTA   = 0.1
C       = 1.0

T_MAX      = 100
ARENA_R    = 1.0       # normalised unit circle
PLAYBACK_FPS = 30
ANIM_INTERVAL = 1000 // PLAYBACK_FPS

# ── Colours ───────────────────────────────────────────────────────────────────
COL = dict(
    S = "#4fc3f7",   # sky blue
    I = "#ef5350",   # red
    R = "#66bb6a",   # green
    D = "#78909c",   # grey
    P = "#ffa726",   # amber  – predators
)

# ── ODE solver ────────────────────────────────────────────────────────────────
def odes(t, y):
    S, I, R, P, D = y
    S, I, R, P = max(S, 0), max(I, 0), max(R, 0), max(P, 0)
    N = max(S + I + R, 1e-6)
    dS = MU*N      - BETA*S*I/N       - PHI*P*S
    dI = BETA*S*I/N - GAMMA*I - EPSILON*I - PHI*C*P*I
    dR = GAMMA*I   - PHI*P*R
    dP = ALPHA*PHI*P*N - DELTA*P
    dD = EPSILON*I + PHI*P*N
    return [dS, dI, dR, dP, dD]

sol = solve_ivp(
    odes, [0, T_MAX],
    [NUM_SUSCEPTIBLE, NUM_INFECTED, NUM_RECOVERED, NUM_PREDATOR, 0],
    method='RK45', max_step=0.1, dense_output=True,
    t_eval=np.linspace(0, T_MAX, 500),
)
T_vals = sol.t
S_vals, I_vals, R_vals, P_vals, D_vals = sol.y

# ── Static agent positions (placed once, never move) ─────────────────────────
rng = np.random.default_rng(42)
N_DOTS = TOTAL_AGENTS + NUM_PREDATOR   # max possible dots needed

angles  = rng.uniform(0, 2*np.pi, N_DOTS)
radii   = np.sqrt(rng.uniform(0, 1, N_DOTS)) * ARENA_R * 0.92
dot_x   = radii * np.cos(angles)
dot_y   = radii * np.sin(angles)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6), facecolor="#0d1117")
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], figure=fig)
gs.update(left=0.04, right=0.97, top=0.88, bottom=0.12, wspace=0.08)

ax_sim = fig.add_subplot(gs[0])
ax_sim.set_facecolor("#0d1117")
ax_sim.set_aspect('equal')
ax_sim.set_xlim(-1.12, 1.12)
ax_sim.set_ylim(-1.12, 1.12)
ax_sim.axis('off')

# Arena circle
theta_c = np.linspace(0, 2*np.pi, 300)
ax_sim.plot(np.cos(theta_c), np.sin(theta_c), color="#30363d", lw=1.5, zorder=0)

ax_line = fig.add_subplot(gs[1])
ax_line.set_facecolor("#0d1117")
for spine in ax_line.spines.values():
    spine.set_color("#30363d")
ax_line.tick_params(colors="#8b949e", labelsize=8)
ax_line.set_xlabel("Time", color="#8b949e", fontsize=9)
ax_line.set_ylabel("Population", color="#8b949e", fontsize=9)
ax_line.set_xlim(0, T_MAX)
ax_line.set_ylim(0, max(S_vals.max(), 10) * 1.05)

# Background line traces
ax_line.plot(T_vals, S_vals, color=COL['S'], lw=1, alpha=0.25)
ax_line.plot(T_vals, I_vals, color=COL['I'], lw=1, alpha=0.25)
ax_line.plot(T_vals, R_vals, color=COL['R'], lw=1, alpha=0.25)
ax_line.plot(T_vals, D_vals, color=COL['D'], lw=1, alpha=0.25)
ax_line.plot(T_vals, P_vals, color=COL['P'], lw=1, alpha=0.25)

# Live line markers (dots that travel along each curve)
live_dots = {
    k: ax_line.plot([], [], 'o', color=COL[k], ms=5, zorder=5)[0]
    for k in ('S','I','R','D','P')
}
# Filled area under I curve for drama
ax_line.fill_between(T_vals, I_vals, alpha=0.06, color=COL['I'])

# Vertical time cursor
time_line = ax_line.axvline(0, color="#58a6ff", lw=0.8, alpha=0.6, zorder=4)

# Legend
legend_handles = [
    Line2D([0],[0], marker='o', color='none', markerfacecolor=COL[k],
           markersize=7, label=lbl)
    for k, lbl in [('S','Susceptible'),('I','Infected'),('R','Recovered'),
                   ('D','Dead'),('P','Predator')]
]
ax_line.legend(handles=legend_handles, loc='upper right',
               facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=8)

# Title
fig.text(0.5, 0.95, "SIR + Predator Model", ha='center', va='top',
         color="#e6edf3", fontsize=14, fontweight='bold',
         fontfamily='monospace')

# ── Scatter for agent dots ────────────────────────────────────────────────────
scat = ax_sim.scatter(dot_x, dot_y, s=3, c=COL['S'],
                      linewidths=0, alpha=0.85, zorder=2)

# Time label on sim panel
time_txt = ax_sim.text(0, -1.07, "t = 0.00", ha='center', va='top',
                       color="#8b949e", fontsize=8, fontfamily='monospace')

# ── Animation update ──────────────────────────────────────────────────────────
COLORS_RGBA = {
    'S': [0.31, 0.76, 0.97, 0.85],
    'I': [0.94, 0.33, 0.31, 0.85],
    'R': [0.40, 0.73, 0.41, 0.85],
    'D': [0.47, 0.56, 0.61, 0.85],
    'P': [1.00, 0.65, 0.15, 0.85],
}

def update(frame):
    s = max(int(round(S_vals[frame])), 0)
    i = max(int(round(I_vals[frame])), 0)
    r = max(int(round(R_vals[frame])), 0)
    d = max(int(round(D_vals[frame])), 0)
    p = max(int(round(P_vals[frame])), 0)

    colour_array = np.zeros((N_DOTS, 4))
    idx = 0
    for k, count in [('S',s),('I',i),('R',r),('D',d),('P',p)]:
        end = min(idx + count, N_DOTS)
        colour_array[idx:end] = COLORS_RGBA[k]
        idx = end
    scat.set_facecolor(colour_array)

    t = T_vals[frame]
    time_txt.set_text(f"t = {t:.1f}")
    time_line.set_xdata([t, t])

    for k, arr in [('S', S_vals),('I', I_vals),('R', R_vals),
                   ('D', D_vals),('P', P_vals)]:
        live_dots[k].set_data([t], [arr[frame]])

    return scat, time_txt, time_line, *live_dots.values()

ani = FuncAnimation(fig, update, frames=len(T_vals),
                    interval=ANIM_INTERVAL, blit=True)

plt.show()