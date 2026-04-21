import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ModelPresets import PRESETS, PRESET_ICS, PRESET_TMAX, DEFAULT_ICS

# ── Shared initial conditions ─────────────────────────────────────────────────
NUM_SUSCEPTIBLE = 999
T_MAX           = 1000

# ── ODE definition (6 variables: S, I, R, P, D, CI) ──────────────────────────
def make_odes(p):
    def odes(t, y):
        S, I, R, P, D, CI = y                         # unpack 6 variables
        S, I, R, P = max(S, 0), max(I, 0), max(R, 0), max(P, 0)
        N = max(S + I + R, 1e-6)
        new_infections = p['beta']*S*I/N
        dS  = p['mu']*N - new_infections - p['phi']*P*S
        dI  = new_infections - p['gamma']*I - p['epsilon']*I - p['phi']*p['c']*P*I
        dR  = p['gamma']*I - p['phi']*P*R
        dP  = p['alpha']*p['phi']*P*N - p['delta']*P
        dD  = p['epsilon']*I
        dCI = new_infections                           # only the inflow, never decremented
        return [dS, dI, dR, dP, dD, dCI]
    return odes

def run_preset(name, p):
    ics = {**DEFAULT_ICS, **PRESET_ICS.get(name, {})}
    i0, p0 = ics['i0'], ics['p0']
    s0 = max(NUM_SUSCEPTIBLE - i0, 0)
    t_max = PRESET_TMAX.get(name, 100)

    sol = solve_ivp(
        make_odes(p), [0, t_max],
        [s0, i0, 0, p0, 0, i0],               # CI starts at i0 (index case)
        method='RK45', max_step=0.1,
        t_eval=np.linspace(0, t_max, t_max * 10)
    )
    return sol.t, sol.y

# ── Plot ──────────────────────────────────────────────────────────────────────
N_PRESETS = len(PRESETS)
NCOLS     = 3
NROWS     = int(np.ceil(N_PRESETS / NCOLS))

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(15, NROWS * 4))
fig.suptitle("SIR + Predator Model — All Presets", fontsize=14, fontweight='bold', y=1.01)
axes_flat = axes.flatten()

plt.tight_layout()

infected_totals = []
disease_deaths  = []
peak_infected   = []

for ax, (name, params) in zip(axes_flat, PRESETS.items()):
    T, (S, I, R, P, D, CI) = run_preset(name, params)   # unpack 6 arrays

    dt             = T[1] - T[0]
    steps_per_unit = int(round(1.0 / dt))

    peak_idx  = np.argmax(I)
    post_peak = I[peak_idx:]
    zero_mask = post_peak < 1.0
###
#    if zero_mask.any() and I[peak_idx] > 10.0:
#        zero_idx = peak_idx + np.argmax(zero_mask)
#        end_idx  = min(zero_idx + 5 * steps_per_unit, len(T))
#    else:
#        end_idx  = len(T)
###
    end_idx  = len(T)
    T_plot  = T[:end_idx]
    S_plot, I_plot, R_plot, P_plot, D_plot, CI_plot = (
        arr[:end_idx] for arr in (S, I, R, P, D, CI)
    )

    N_plot  = S_plot + I_plot + R_plot

    ax.plot(T_plot, S_plot,  color='#4fc3f7', lw=1.5, label='Susceptible')
    ax.plot(T_plot, I_plot,  color='#ef5350', lw=1.5, label='Infected')
    ax.plot(T_plot, R_plot,  color='#66bb6a', lw=1.5, label='Recovered')
    ax.plot(T_plot, D_plot,  color='#78909c', lw=1.5, label='Dead')
    ax.plot(T_plot, P_plot,  color='#ffa726', lw=1.5, label='Predator')

    total_dead     = D_plot[-1]
    total_infected = CI_plot[-1]

    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time', fontsize=8)
    ax.set_ylabel('Population', fontsize=8)
    ax.set_xlim(0, T_plot[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.97, 0.97,
            f'Total infected: {total_infected:.0f}\nTotal dead: {total_dead:.0f}',
            transform=ax.transAxes, fontsize=8,
            va='top', ha='right', color='#78909c',
            clip_on=False,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#78909c', alpha=0.7))
    infected_totals.append(CI_plot[-1])
    disease_deaths.append(D_plot[-1])
    peak_infected.append(np.max(I_plot))

for ax in axes_flat[N_PRESETS:]:
    ax.set_visible(False)

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5,
           fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 5))
fig2.suptitle("Epidemic Summary by Preset", fontweight='bold')

names = list(PRESETS.keys())
for ax, vals, title in zip(axes2,
                           [infected_totals, disease_deaths, peak_infected],
                           ['Total ever infected', 'Total disease deaths', 'Peak simultaneous infected']):
    ax.bar(names, vals, color='#ef5350')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig2.tight_layout()

plt.show()