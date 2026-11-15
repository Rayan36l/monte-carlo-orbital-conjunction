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

# TASK 5/ Optimized Score Function: caches trajectory states and checks all roots
def score_optimized(
    step_function,
    dynamics_function,
    t0,
    state1_0,
    state2_0,
    t_start,
    t_end,
    h,
    tol=1e-6,
    max_iter=100,
    return_details=False,
):
    """
    Optimized conjunction score function that propagates once from t0 to t_end,
    reuses cached bracket endpoints during bisection, and checks all roots
    of D(t) = 0 plus window boundaries to return the global minimum distance.
    """
    times, states1 = propagate(step_function, dynamics_function, t0, state1_0, t_end, h)
    _, states2 = propagate(step_function, dynamics_function, t0, state2_0, t_end, h)

    i_start = np.searchsorted(times, t_start)
    i_end = np.searchsorted(times, t_end)
    if i_start >= len(times):
        raise ValueError("t_start is outside the propagated time array.")
    if i_end >= len(times):
        i_end = len(times) - 1

    D_values = np.zeros(len(times))
    for i in range(i_start, i_end + 1):
        D_values[i] = evaluate_D(states1[i], states2[i])

    brackets = []
    for i in range(i_start, i_end):
        D_left = D_values[i]
        D_right = D_values[i + 1]
        if D_left == 0.0:
            brackets.append((times[i], times[i], i))
        elif D_left * D_right < 0.0:
            brackets.append((times[i], times[i + 1], i))

    def D_of_t_cached(t, left_index):
        t_left = times[left_index]
        state1_left = states1[left_index]
        state2_left = states2[left_index]
        _, local_states1 = propagate(step_function, dynamics_function, t_left, state1_left, t, h)
        _, local_states2 = propagate(step_function, dynamics_function, t_left, state2_left, t, h)
        return evaluate_D(local_states1[-1], local_states2[-1])

    root_times = []
    total_iters = 0
    for t_low, t_high, left_index in brackets:
        if t_low == t_high:
            root_times.append(t_low)
            continue

        def local_D(t):
            return D_of_t_cached(t, left_index)

        root, iters = bisection(local_D, t_low, t_high, tol=tol, max_iter=max_iter)
        root_times.append(root)
        total_iters += iters

    candidate_times = [t_start, t_end]
    candidate_times.extend(root_times)

    def distance_at_time(t_candidate):
        left_index = np.searchsorted(times, t_candidate, side="right") - 1
        left_index = max(0, min(left_index, len(times) - 1))
        t_left = times[left_index]
        state1_left = states1[left_index]
        state2_left = states2[left_index]
        _, local_states1 = propagate(step_function, dynamics_function, t_left, state1_left, t_candidate, h)
        _, local_states2 = propagate(step_function, dynamics_function, t_left, state2_left, t_candidate, h)
        _, _, distance = relative_state(local_states1[-1], local_states2[-1])
        return distance

    candidate_distances = np.array([distance_at_time(t_c) for t_c in candidate_times])
    best_index = int(np.argmin(candidate_distances))
    t_tca = candidate_times[best_index]
    d_min = float(candidate_distances[best_index])

    if return_details:
        return {
            "t_tca": t_tca,
            "d_min": d_min,
            "total_iters": total_iters,
            "root_times": np.array(root_times),
            "candidate_times": np.array(candidate_times),
            "candidate_distances": candidate_distances,
            "times": times,
            "states1": states1,
            "states2": states2,
            "D_values": D_values,
            "brackets": brackets,
        }

    return (t_tca, d_min, total_iters)
