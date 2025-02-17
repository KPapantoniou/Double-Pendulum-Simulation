import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation





def double_pendulumm_ode(t, y, m1, m2, l1, l2, g):
    """
    ODE function for a double pendulum.
    Args:
        t: time
        y: state vector [theta1, theta2, omega1, omega2]
        m1: mass of the first pendulum
        m2: mass of the second pendulum
        l1: length of the first pendulum
        l2: length of the second pendulum
        g: acceleration due to gravity
    Returns:
        derivative of the state vector
    """
    theta1, omega1, theta2, omega2 = y
    delta =theta2 - theta1
    
    ## Matrix Ax = b
    ## A components
    
    a11 = (m1+m2)*l1
    a12 = m2*l2*np.cos(delta)
    a21 = l1*np.cos(delta)
    a22 = l2
    
    ## b components
    
    b1 = m2*l2*omega2**2*np.sin(delta) - (m1+m2)*g*np.sin(theta1)
    b2 = -l1*omega1**2*np.sin(delta) - g*np.sin(theta2)
    
    ## Determinant of A
    det = a11*a22 - a12*a21
    
    ##Crammer's rule to solve the system of equations
    
    alpha1 = (b1*a22 - b2*a12)/det
    alpha2 = (a11*b2 - a21*b1)/det
    
    return [omega1, alpha1, omega2, alpha2]

def calculate_energy(theta1, theta2, omega1, omega2, L1, L2, m1, m2, g):
    """
    Calculate kinetic and potential energy of the system.
    Args:
        theta1, theta2: Angles of the pendulums
        omega1, omega2: Angular velocities
        L1, L2: Lengths of the pendulums
        m1, m2: Masses of the pendulums
        g: Acceleration due to gravity
    Returns:
        KE: Kinetic energy
        PE: Potential energy
    """
    ## Kinetic energy
    v1_sq = (L1 * omega1)**2
    v2_sq = (L1 * omega1)**2 + (L2 * omega2)**2 + 2 * L1 * L2 * omega1 * omega2 * np.cos(theta2 - theta1)
    KE = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq

    ## Potential energy
    PE = -m1*g*L1*np.cos(theta1) - m2*g*(L1*np.cos(theta1+L2*np.cos(theta2)))
    return KE, PE

def run_simulation(m1, m2, L1, L2, g, theta1_initial, omega1_initial, theta2_initial, omega2_initial, time=30, time_step=0.01):
    # Initial conditions
    y0 = [np.radians(theta1_initial), omega1_initial, np.radians(theta2_initial), omega2_initial]
    time_eval = np.arange(0, time, time_step)
    ##Solve ODE

    solution = solve_ivp(
        double_pendulumm_ode,
        [0,time],
        y0,
        args=(L1,L2,m1,m2,g),
        t_eval=time_eval,
        rtol = 1e-8     
    )

    theta1 = solution.y[0]
    theta2 = solution.y[2]
    omega1 = solution.y[1]
    omega2 = solution.y[3]

    ##Coordinates
    x1 = L1*np.sin(theta1)
    y1 = -L1*np.cos(theta1)

    x2 = x1 + L2*np.sin(theta2)
    y2 = y1 - L2*np.cos(theta2)

    KE, PE = calculate_energy(theta1, theta2, solution.y[1], solution.y[3], L1, L2, m1, m2, g)
    total_energy = KE + PE
    
    return time_eval, theta1, theta2, KE, PE, total_energy, x1, y1, x2, y2, omega1, omega2


    


if __name__ == "__main__":  
    m1 = 2.0
    m2 = 2.0
    L1 = 1.0
    L2 = 1.0
    g = 9.81
    theta1_initial = 10
    theta2_initial = 130
    omega1_initial =0
    omega2_initial = 0
    time =30
    time_step = 0.01
    
    time_eval, theta1, theta2, KE, PE, total_energy, x1, y1, x2, y2, omega1, omega2 = run_simulation(m1, m2, L1, L2, g, theta1_initial, omega1_initial, theta2_initial, omega2_initial, time, time_step)
    
    ##Plot Angles over Time
    plt.figure(figsize=(12,6))
    plt.plot(time_eval, theta1, label='θ1 (rad)',color='b')
    plt.plot(time_eval,theta2, label='θ2 (rad)', color = 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (radians)')
    plt.title('Double Pendulum: Angular Motion')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12,6))
    plt.plot(omega1, theta1, label='θ1 (rad)')
    plt.xlabel('Omega (rad/s)')
    plt.ylabel('Angle (radians)')
    plt.title('Double Pendulum: Angular Motion')
    plt.legend()
    plt.grid()
    plt.show()

    ##Animate Double Pendulum

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-2.5,2.5)
    ax.set_ylim(-2.5,2.5)
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2, markersize=8, label="Pendulum Motion")
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([],[])
        time_text.set_text('')
        return line, time_text 

    def animate(i):
        line.set_data([0,x1[i],x2[i]],[0,y1[i],y2[i]])
        time_text.set_text(f'Time = {time_eval[i]:.2f}s')
        return line, time_text 

    ani = FuncAnimation(
        fig,
        animate,
        frames=len(time_eval),
        init_func=init,
        blit=True,
        interval=time_step*1000
    )
    plt.legend()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_xlim(min(theta1) - 0.1, max(theta1) + 0.1)
    ax2.set_ylim(min(theta2) - 0.1, max(theta2) + 0.1)
    ax2.set_xlabel('θ1 (rad)')
    ax2.set_ylabel('θ2 (rad)')
    ax2.set_title('Phase Space: θ1 vs. θ2')
    ax2.grid()

    line2, = ax2.plot([],[],'b-',lw=1)
    point, = ax2.plot([],[],'ro')

    def init_phase():
        line2.set_data([], [])
        point.set_data([], [])
        return line2, point

    def animate_phase(i):
        line2.set_data(theta1[:i], theta2[:i])
        point.set_data([theta1[i]], [theta2[i]])
        return line2, point

    ani2 = FuncAnimation(
        fig2, animate_phase, frames=len(time_eval), init_func=init_phase,
        blit=True, interval=time_step*1000
    )

    plt.show()




    plt.figure(figsize=(12, 6))
    plt.plot(time_eval, KE, label='Kinetic Energy')
    plt.plot(time_eval, PE, label='Potential Energy')
    plt.plot(time_eval, total_energy, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (Joules)')
    plt.title('Energy Conservation in Double Pendulum')
    plt.legend()
    plt.grid()
    plt.show()