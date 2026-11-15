import numpy as np
from orbital.integrators import propagate


# TASK 1/ Relative State Function: computes the relative position, relative velocity, and distance between two objects.
def relative_state(state1, state2):
    r1 = state1[:3] # :3 =  first three elemenets (x, y, z)
    v1 = state1[3:] # 3: = last three elements (vx, vy, vz)
    r2 = state2[:3] 
    v2 = state2[3:]
    r_rel = r1 - r2
    v_rel = v1 - v2
    d = np.linalg.norm(r_rel)
    return r_rel, v_rel, d

'''TASK 2/ Bisection Root Finder: narrows down the time of closest approach by repeatedly halving an interval until 
D(t) = 0 (TCA) is found within a given tolerance.'''
def bisection(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) > 0:
        raise ValueError("No sign change")

    for i in range(max_iter):
        c = (a + b) / 2
        c_eval = f(c)
        if f(a) * c_eval < 0:
            b = c
        else:
            a = c
        if abs(b - a) < tol:
            return c, i + 1  
    return c, max_iter

# TASK 3/ evaluate_D: Computing approach indicator from two state vectors at the same time
def evaluate_D(state1, state2):
    r_rel, v_rel, d = relative_state(state1, state2)
    D = np.dot(r_rel, v_rel)
    return D 

# TASK 4/ SCORE Function: Computing the time of closest approach and miss distance for a conjunction event.
def score(step_function, dynamics_function, t0, state1_0, state2_0, t_start, t_end, h, tol=1e-6):
    # Step 1: Propagating both objects from t0 to t_end
    times, states1 = propagate(step_function, dynamics_function, t0, state1_0, t_end, h)
    _, states2 = propagate(step_function, dynamics_function, t0, state2_0, t_end, h) # _, means I don't need the times because I already got them

    # Step 2: Computing D at every saved time step
    i_start = np.searchsorted(times, t_start) # finding the index
    D_values = np.zeros(len(times)) # defines what D_values represents (like creating an empty table before filling it in)
    for i in range(i_start, len(times)):
        D_values[i] = evaluate_D(states1[i], states2[i])

    # Step 3: Finding the first negative-to-positive bracket
    t_low = None
    for i in range(i_start, len(times) - 1):
        if D_values[i] < 0 and D_values[i+1] > 0:
            t_low = times[i]
            t_high = times[i+1]
            break

    if t_low is None:
        raise ValueError("No valid bracket found")
    
    # Step 4: Definng a function that evaluates D at any time t
    def D_of_t(t): 
        _, s1 = propagate(step_function, dynamics_function, t0, state1_0, t, h)
        _, s2 = propagate(step_function, dynamics_function, t0, state2_0, t, h)
        return evaluate_D(s1[-1], s2[-1])

    # Step 5
    t_tca, iters = bisection(D_of_t, t_low, t_high, tol)  

    # Step 6
    _, s1 = propagate(step_function, dynamics_function, t0, state1_0, t_tca, h)
    _, s2 = propagate(step_function, dynamics_function, t0, state2_0, t_tca, h)
    _, _, d_min = relative_state(s1[-1], s2[-1])

    return (t_tca, d_min, iters)  
