import optas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Planner:

    def __init__(self):

        # Planner attributes
        dt = 0.1  # time step
        obs = [0, 0]  # obstacle position
        obs_rad = 0.9  # obstacle radii

        # Setup rectangle model
        pm_width = 0.32  # half width of rectangle
        pm_height = 0.16  # height of rectangle
        pm_radius = (pm_width**2 + pm_height**2)**(0.5)   #calculating radius at corners
        pm_dim = 2  # x, y dimensions
        dlim = {0: [-1.5, 1.5], 1: [-1.5, 1.5]}  # pos/vel limits
        rectangle = optas.TaskModel('rectangle', pm_dim, time_derivs=[0, 1], dlim=dlim)
        # rectangle.set_bounds([-pm_width, -pm_height], [pm_width, pm_height])
        pm_name = rectangle.get_name()

        # Setup optimization builder
        T = 45 # number of time steps
        builder = optas.OptimizationBuilder(T, tasks=rectangle, derivs_align=True)

        # Add parameters
        init = builder.add_parameter('init', 2)  # initial point mass position
        goal = builder.add_parameter('goal', 2)  # goal point mass position

        # Constraint: limits
        builder.enforce_model_limits(pm_name, time_deriv=0)
        builder.enforce_model_limits(pm_name, time_deriv=1)

        # Constraint: dynamics
        builder.integrate_model_states(pm_name, time_deriv=1, dt=dt)

        # Constraint: initial state
        builder.fix_configuration(pm_name, config=init)
        builder.fix_configuration(pm_name, time_deriv=1)

        # Constraint: final velocity
        dxF = builder.get_model_state(pm_name, -1, time_deriv=1)
        builder.add_equality_constraint('final_velocity', dxF)

        # Constraint: obstacle avoidance
        X = builder.get_model_states(pm_name)
        safe_dist_sq = (obs_rad + pm_radius)**2
        for i in range(T):
            dist_sq = optas.sumsqr(obs - X[:, i])
            builder.add_geq_inequality_constraint(f'obs_avoid_{i}', dist_sq, safe_dist_sq)

        # Cost: final state
        builder.add_cost_term('final_state', optas.sumsqr(goal - X[:, -1]))

        # Cost: minimize velocity
        w = 0.01/float(T)  # weight on cost term
        dX = builder.get_model_states(pm_name, time_deriv=1)
        builder.add_cost_term('minimize_velocity', w*optas.sumsqr(dX))

        # Cost: minimize acceleration
        w = 0.005/float(T)  # weight on cost term
        ddX = (dX[:, 1:] - dX[:, :-1])/dt
        builder.add_cost_term('minimize_acceleration', w*optas.sumsqr(ddX))

        # Create solver
        self.solver = optas.CasADiSolver(builder.build()).setup('ipopt')

        # Save variables
        self.T = T
        self.dt = dt
        self.pm_name = pm_name
        self.pm_radius = pm_radius
        self.pm_width = pm_width
        self.pm_height = pm_height
        self.obs = obs
        self.obs_rad = obs_rad
        self.duration = float(T-1)*dt  # task duration
        self.rectangle = rectangle
        self.solution = None

    def plan(self, init, goal):
        self.solver.reset_parameters({'init': init, 'goal': goal})
        solution = self.solver.solve()
        plan_x = self.solver.interpolate(solution[f'{self.pm_name}/x'], self.duration)
        plan_dx = self.solver.interpolate(solution[f'{self.pm_name}/dx'], self.duration)
        return plan_x, plan_dx

class Animate:

    def __init__(self, animate, width=0.4, height=0.2):

        # Setup planner
        self.planner = Planner()
        self.init = [-1.5, -1.5]
        self.rectangle_init = [self.init[0] - self.planner.pm_width, self.init[1] - self.planner.pm_height]
        self.goal = [1, 1]
        self.rectangle_goal = [self.goal[0] - self.planner.pm_width, self.goal[1] - self.planner.pm_height]
        self.plan_x, self.plan_dx = self.planner.plan(self.init, self.goal)

        # Setup figure
        self.t = optas.np.linspace(0, self.planner.duration, self.planner.T)
        self.X = self.plan_x(self.t)
        self.dX = self.plan_dx(self.t)

        self.fig, self.ax = plt.subplot_mosaic([['birdseye', 'position'],
                                                ['birdseye', 'velocity']],
                                               layout='constrained',
                                               figsize=(10, 5),
                                                )
        self.mpc_line, = self.ax['birdseye'].plot([], [], '-x', color='yellow', label='mpc')
        self.ax['birdseye'].plot(self.X[0, :], self.X[1, :], '-kx', label='plan')
        self.ax['birdseye'].add_patch(plt.Rectangle(self.rectangle_init, width=2*self.planner.pm_width, height=2*self.planner.pm_height, color='green', alpha=0.5))
        self.ax['birdseye'].add_patch(plt.Rectangle(self.rectangle_goal, width=2*self.planner.pm_width, height=2*self.planner.pm_height, color='red', alpha=0.5))
        self.dt = self.planner.dt
        self.obs_pos = optas.np.array(self.planner.obs)
        self.obs_visual = plt.Rectangle(xy=self.obs_pos, width=2*self.planner.pm_width, height=2*self.planner.pm_height, angle=15.0, color='black') #Circle(self.obs_pos, radius=self.planner.obs_rad, color='black') # 
        self.ax['birdseye'].add_patch(self.obs_visual)
        self.ax['birdseye'].set_aspect('equal')
        self.ax['birdseye'].set_xlim(*self.planner.rectangle.dlim[0])
        self.ax['birdseye'].set_ylim(*self.planner.rectangle.dlim[0])
        self.ax['birdseye'].set_title('Birdseye View')
        self.ax['birdseye'].set_xlabel('x')
        self.ax['birdseye'].set_ylabel('y')

        self.ax['position'].plot(self.t, self.X[0,:], '-rx', label='plan-x')
        self.ax['position'].plot(self.t, self.X[1,:], '-bx', label='plan-y')
        self.pm_pos_curr_x, = self.ax['position'].plot([], [], 'or', label='curr-x')
        self.pm_pos_curr_y, = self.ax['position'].plot([], [], 'ob', label='curr-y')
        self.ax['position'].set_ylabel('Position')
        self.ax['position'].set_xlim(0, self.planner.duration)

        axlim = max([abs(l) for l in self.planner.rectangle.dlim[0]])
        self.ax['position'].set_ylim(-axlim, axlim)

        self.ax['velocity'].plot(self.t, self.dX[0,:], '-rx', label='plan-dx')
        self.ax['velocity'].plot(self.t, self.dX[1,:], '-bx', label='plan-dy')
        self.pm_vel_curr_x, = self.ax['velocity'].plot([], [], 'or', label='curr-dx')
        self.pm_vel_curr_y, = self.ax['velocity'].plot([], [], 'ob', label='curr-dy')
        self.ax['velocity'].axhline(self.planner.rectangle.dlim[1][0], color='red', linestyle='--')
        self.ax['velocity'].axhline(self.planner.rectangle.dlim[1][1], color='red', linestyle='--', label='limit')
        self.ax['velocity'].set_ylabel('Velocity')
        self.ax['velocity'].set_xlabel('Time')

        self.ax['velocity'].set_xlim(0, self.planner.duration)
        axlim = max([abs(1.5*l) for l in self.planner.rectangle.dlim[1]])
        self.ax['velocity'].set_ylim(-axlim, axlim)

        for a in self.ax.values():
            a.legend(ncol=3, loc='lower right')
            a.grid()

        # Animate
        if not animate: return
        self.pos_line = self.ax['position'].axvline(color='blue', alpha=0.5)
        self.vel_line = self.ax['velocity'].axvline(color='blue', alpha=0.5)
        self.pm_visual = plt.Rectangle(xy=self.rectangle_init, width=2*self.planner.pm_width, height=2*self.planner.pm_height, angle=0.0, color='black')  # Circle(self.init, radius=self.planner.pm_radius, color='blue', alpha=0.5)
        self.frames = list(range(self.planner.T))
        self.ani = FuncAnimation(self.fig, self.update, frames=self.frames, blit=True)

        

    def update(self, frame):

        # Udpate position/velocity indicator line
        self.pos_line.set_xdata([frame, frame])
        self.vel_line.set_xdata([frame, frame])
        # Update point mass
        self.pm_visual.set_xy(self.plan_x(frame) - (self.planner.pm_width, self.planner.pm_height))
        self.ax['birdseye'].add_patch(self.pm_visual)
        return (self.pm_visual, self.pos_line, self.vel_line)

    @staticmethod
    def show():
        plt.show()

def main():
    from sys import argv
    animate = '--noanimate' not in argv
    Animate(animate).show()

if __name__ == '__main__':
    main()
