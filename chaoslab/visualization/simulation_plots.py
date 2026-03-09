import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_simulation_diagnostics(t, theta1, theta2, omega1, energy, x1, y1, x2, y2, show=True):
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(t, theta1, label="theta1")
    ax1.plot(t, theta2, label="theta2")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Angle [rad]")
    ax1.set_title("Double Pendulum Angles")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(theta1, omega1, lw=0.8)
    ax2.set_xlabel("theta1 [rad]")
    ax2.set_ylabel("omega1 [rad/s]")
    ax2.set_title("Phase Portrait: theta1 vs omega1")
    ax2.grid(True, alpha=0.3)

    fig3, ax3 = plt.subplots(figsize=(11, 4))
    ax3.plot(t, energy, label="Total energy")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Energy")
    ax3.set_title("Energy Consistency Check")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig4, ax4 = plt.subplots(figsize=(6, 6))
    ax4.plot([0, x1[0], x2[0]], [0, y1[0], y2[0]], "o-", label="t=0")
    ax4.plot([0, x1[-1], x2[-1]], [0, y1[-1], y2[-1]], "o-", label="t=end")
    ax4.set_xlim(-2.5, 2.5)
    ax4.set_ylim(-2.5, 2.5)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_title("Pendulum Configuration (start/end)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig1, fig2, fig3, fig4


def animate_pendulum(t, x1, y1, x2, y2, interval_ms=None, show=True):
    if interval_ms is None:
        if len(t) >= 2:
            interval_ms = float((t[1] - t[0]) * 1000.0)
        else:
            interval_ms = 20.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Double Pendulum Animation")

    line, = ax.plot([], [], "o-", lw=2, markersize=7)
    time_text = ax.text(0.03, 0.95, "", transform=ax.transAxes)

    def _init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def _update(i):
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        time_text.set_text(f"t = {t[i]:.2f}s")
        return line, time_text

    ani = FuncAnimation(
        fig,
        _update,
        frames=len(t),
        init_func=_init,
        interval=interval_ms,
        blit=True,
    )

    if show:
        plt.show()

    return fig, ani
