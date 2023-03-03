import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.plot([i], [i], 'ro')

ani = FuncAnimation(fig, animate, frames=10, interval=500, blit=False)
plt.show()