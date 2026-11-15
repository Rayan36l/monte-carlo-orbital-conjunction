import numpy as np


# Task 1
def rk4(dynamics_function, t, y, h):
    k1 = dynamics_function(t, y)
    k2 = dynamics_function(t + h/2, y + h/2 * k1)
    k3 = dynamics_function(t + h/2, y + h/2 * k2)
    k4 = dynamics_function(t + h, y + h * k3)
    y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return  y_next


def propagate(step_function, dynamics_function, t0, y0, tf, h):
    t = t0
    y = y0.copy()
    times = [t0]
    states = [y.copy()]
    
    while t < tf:
        h = min(h, tf - t) # comparing h and delta t, whichever is smaller becomes the new h, just to prevent overshooting tf
        y = step_function(dynamics_function, t, y, h)
        t = t + h # h is the time step size (how many seconds you advance forward in each iteration of the loop)
        times.append(t)
        states.append(y.copy()) # the append command adds new stuff to the list
    return np.array(times), np.array(states)



