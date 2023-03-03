import optas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Planner:

    def __init__(self):

        # Planner attributes
        dt = 0.1  # time step
        obs = [0, 0]  # obstacle position
        obs_rad = 0.35  # obstacle radii

        # Setup point mass model
        pm_radius = 0.32  # point mass radii
        pm_dim = 2 # x, y dimensions
        dlim = {0: [-1.5, 1.5], 1: [-1.5, 1.5]}  # pos/vel limits
        point_mass = optas.TaskModel('point_mass', pm_dim, time_derivs=[0, 1], dlim=dlim)
        pm_name = point_mass.get_name()

        # Setup optimization builder
        T = 60 # number of time steps
        builder = optas.OptimizationBuilder(T, tasks=point_mass, derivs_align=True)

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
        self.solver = optas.CasADiSolver(builder.build()).setup('ipopt')

        # Save variables
        self.T = T
        self.dt = dt
        self.pm_name = pm_name
        self.pm_radius = pm_radius
        self.obs = obs
        self.obs_rad = obs_rad
        self.duration = float(T-1)*dt  # task duration
        self.point_mass = point_mass

    def plan(self, init, goal):
        self.solver.reset_parameters({'init': init, 'goal': goal})
        solution = self.solver.solve()
        plan_x = self.solver.interpolate(solution[f'{self.pm_name}/x'], self.duration)
        plan_dx = self.solver.interpolate(solution[f'{self.pm_name}/dx'], self.duration)
        return plan_x, plan_dx