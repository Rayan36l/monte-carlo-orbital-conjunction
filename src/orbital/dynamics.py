import numpy as np

# Task 3
def two_body(t, state, mu = 398600.4415):
    x, y, z, vx, vy, vz = state # unpacking the state vector just to avoid using index numbers like state[0] etc. Also, python knows state is a numpy array, so there's immediate index.
    r = np.sqrt(x**2 + y**2 + z**2) # just for clarification, this is the distance from Earth's center to the satellite.
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3
    return np.array([vx, vy, vz, ax, ay, az]) # derivative of state vector (hence the velocities and accelerations)
