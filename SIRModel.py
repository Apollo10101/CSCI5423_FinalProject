from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

NUM_SUSCEPTIBLE = 999
NUM_INFECTED    = 1
NUM_PREDATOR    = 0

BETA    = 5.0
GAMMA   = 0.5
EPSILON = 0.1
PHI     = 0.1
MU      = 0.05
ALPHA   = 0.05
DELTA   = 0.1
C       = 1.0
T_MAX   = 100

def odes(t, y):
    S, I, R, P, D = y
    S, I, R, P = max(S,0), max(I,0), max(R,0), max(P,0)
    N = max(S+I+R, 1e-6)
    dS = MU*N - BETA*S*I/N - PHI*P*S
    dI = BETA*S*I/N - GAMMA*I - EPSILON*I - PHI*C*P*I
    dR = GAMMA*I - PHI*P*R
    dP = ALPHA*PHI*P*N - DELTA*P
    dD = EPSILON*I + PHI*P*N
    return [dS, dI, dR, dP, dD]

sol = solve_ivp(
    odes, [0, T_MAX],
    [NUM_SUSCEPTIBLE, NUM_INFECTED, 0, NUM_PREDATOR, 0],
    method='RK45', max_step=0.1, dense_output=True,
    t_eval=np.linspace(0, T_MAX, 600),
)
T, (S, I, R, P, D) = sol.t, sol.y

fig, ax = plt.subplots()
ax.plot(T, S, color='b', label='Susceptible')
ax.plot(T, I, color='r', label='Infected')
ax.plot(T, R, color='g', label='Recovered')
ax.plot(T, D, color='k', label='Dead')
ax.plot(T, P, color='y', label='Predator')
ax.set_xlabel('time')
ax.set_ylabel('population')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.legend()
plt.show()