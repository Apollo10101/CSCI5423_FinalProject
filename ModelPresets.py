DEFAULT_ICS = dict(i0=1, p0=10)

PRESETS = {
    "Moderate Predation":  dict(beta=3.0, gamma=0.5, epsilon=0.2,  phi=0.1, c=2.0, alpha=0.0,  delta=0.0,  mu=0.1),
    "Predator Elimination":dict(beta=3.0, gamma=0.3, epsilon=0.05, phi=0.3,  c=5.0, alpha=0.02, delta=0.2,  mu=0.3),
    "Endemic Cycling":     dict(beta=2.0, gamma=0.4, epsilon=0.05, phi=0.1,  c=2.0, alpha=0.04, delta=0.15, mu=0.1),
    "Pred-Prey Collapse":  dict(beta=1.5, gamma=0.5, epsilon=0.02, phi=0.4,  c=2.0, alpha=0.1,  delta=0.05, mu=0.4),
    "Disease Extinction":  dict(beta=8.0, gamma=0.1, epsilon=0.4,  phi=0.05, c=1.0, alpha=0.02, delta=0.1,  mu=0.05),
    "Herd Equilibrium":    dict(beta=2.0, gamma=0.8, epsilon=0.05, phi=0.1,  c=1.5, alpha=0.03, delta=0.1,  mu=0.3),
    "No Predation":        dict(beta=3.0, gamma=0.5, epsilon=0.2,  phi=0.0,  c=1.0, alpha=0.0,  delta=0.0,  mu=0.0)
}

PRESET_ICS = {
    "Moderate Predation": dict(p0=5),
    "No Predation": dict(p0=0)
}

PRESET_TMAX = {
    "Moderate Predation":  200,
    "Predator Elimination":200,
    "Endemic Cycling":     1000,
    "Pred-Prey Collapse":  500,
    "Disease Extinction":  100,
    "Herd Equilibrium":    500,
    "No Predation":        100,
}