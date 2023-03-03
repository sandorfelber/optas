import optas
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from optas.spatialmath import *
import math
import numpy as np

def yaw2quat(angle):
    return Quaternion.fromrpy(tr2eul(rotz(angle))).getquat()

def yaw2deg(angle):
    return angle*180/np.pi

# TOMPCCPlanner
class Planner:

    def __init__(self, dt, Lx, Ly):

        # Setup
        mu = 0.1  # coef of friction
        dt = float(dt)  # time step
        nX = 4  # number of state variables
        nU = 4  # number of control variables
        T = 40  # number of step
        Lx = float(Lx)  # length of slider (box) in x-axis
        Ly = float(Ly)  # length of slider (box) in y-axis
        eff_ball_radius = 0.015
        self.Lx = Lx
        self.Ly = Ly
        self.eff_ball_radius = 0.08
        endeffector_location_init = optas.np.array([0.4, 0., 0.06])
        obs_position = [0, 0]  # obstacle position
        obs_rad = 0.25  # radius of obstacle
        self.obs_position = obs_position
        self.obs_rad = obs_rad
        phi_lo = math.atan2(Lx, Ly) # lower limit for phi
        phi_up = -phi_lo # upper limit for phi
        L = optas.diag([1, 1, 0.5])  # limit surface model
        SxC0 = 0.5*Ly # initial contact position in x-axis of slider
        SyC0 = 0. # initial contact position in y-axis of slider
        SphiC0 = 0. # initial contact orientation in slider frame
        self.duration = float(T-1)*dt  # task duration
        Wu = optas.diag([ # weigting on minimizing controls cost term
            0.,  # normal contact force
            0.,  # tangential contact force
            0.,  # angular rate of sliding, positive part
            0.,  # angular rate of sliding, negative part
        ])
        WxT = optas.diag([ # weighting for terminal state cost term
            1., # x-position of slider
            1., # y-position of slider
            0.01, # orientation of slider
            0.001, # orientation of contact
        ])
        we = 0.1  # slack weights
        SphiCT = 0. # final contact orientation in slider frame
    
        rectangle_dim = 2 # x, y dimensions of rectangle
        dlim = {0: [-2, 2], 1: [-1.5, 1.5]}  # pos/vel limits

        # Setup task models
        state = optas.TaskModel('state', dim=nX, time_derivs=[0], dlim=dlim)
        state_alias_for_rectangle = state
        state_name = state.get_name()
        control = optas.TaskModel('control', dim=nU, time_derivs=[0], T=T-1, symbol='u')
        #rectangle_task_model = optas.TaskModel('rectangle_task_model', rectangle_dim, time_derivs=[0, 1], dlim=dlim)
        #rectangle_task_model_name = rectangle_task_model.get_name()

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=T, tasks=[state, control])

        # Add additional decision variables
        eps = builder.add_decision_variables('slack', T-1)

        # Set parameters
        GpS0 = builder.add_parameter('GpS0', 2)  # initial slider position in global frame
        GthetaS0 = builder.add_parameter('GthetaS0') # initial slider orientation in global frame
        GpST = builder.add_parameter('GpST', 2)  # goal slider position in global frame
        GthetaST = builder.add_parameter('GthetaST') # goal slider orientation in global frame

        # Get states/controls
        X = builder.get_model_states('state')
        U = builder.get_model_states('control')

        # Constraint: initial configuration
        x0 = optas.vertcat(GpS0, GthetaS0, SphiC0)
        builder.fix_configuration('state', config=x0)

        # Split X/U
        theta = X[2, :]
        phi = X[3, :]

        fn = U[0,:]
        ft = U[1,:]
        dphip = U[2,:]
        dphim = U[3,:]

        # Constraint: dynamics
        o2 = optas.DM.zeros(2)  # 2-vector of zeros
        o3 = optas.DM.zeros(3)  # 3-vector of zeros
        I = optas.DM.eye(2)  # 2-by-2 identity
        for k in range(T-1):

            # Setup
            xn = X[:, k+1]  # next state
            x = X[:, k]   # current state
            u = U[:, k]  # control input
            R = rotz(theta[k]-0.5*optas.np.pi)  # rotation matrix in xy-plane of slider
            SxC = 0.5*Ly # x-position of box
            SyC = SxC*optas.tan(phi[0]) # y-position of contact
            JC = optas.horzcat(I, optas.vertcat(-SyC, SxC))

            # Compute system dynamics f(x, u) = Ku
            K = optas.vertcat(
                optas.horzcat(R @ L @ JC.T, o3, o3),
                optas.horzcat(o2.T, 1, -1),
            )
            f = K @ u

            # Add constraint
            builder.add_equality_constraint(f'dynamics_{k}', xn, x + dt*f)

        # Constraint: complementarity
        lambda_minus = mu*fn - ft
        lambda_plus = mu*fn + ft
        lambdav = optas.vertcat(lambda_minus, lambda_plus)
        dphiv = optas.vertcat(dphip, dphim)

        builder.add_geq_inequality_constraint('positive_lambdav', lambdav)
        builder.add_geq_inequality_constraint('positive_dphiv', dphiv)

        for k in range(T-1):
            e = eps[k]
            lambdavk = lambdav[:, k]
            dphivk = dphiv[:, k]
            builder.add_equality_constraint(
                f'complementarity_{k}', optas.dot(lambdavk, dphivk) + e,
            )
            
        # Constraint: obstacle avoidance
        #X = builder.get_model_states(rectangle_task_model_name)
        safe_dist_sq = (obs_rad + ((Lx**2 + Ly**2)**0.5))**2
        for i in range(T):
            #print("X[:, i]", X[2:, i])
            dist_sq = optas.sumsqr(obs_position - X[:2, i])  # first two elements of X are position and the second two are velocity
            builder.add_geq_inequality_constraint(f'obs_avoid_{i}', dist_sq, safe_dist_sq)

        # Cost: minimize control magnitude
        for k in range(T-1):
            u = U[:, k]
            builder.add_cost_term(f'min_control_{k}', u.T @ Wu @ u)

        # Cost: terminal state
        xT = optas.vertcat(GpST, GthetaST, SphiCT)
        xbarT = X[:, -1] - xT
        builder.add_cost_term('terminal_state', xbarT.T @ WxT @ xbarT)

        # Cost: slack terms
        builder.add_cost_term('slack', we*cs.sumsqr(eps))

        # Constraint: slack
        builder.add_geq_inequality_constraint('positive_slack', eps)

        # Constraint: bound phi
        builder.add_bound_inequality_constraint('phi_bound', phi_lo, phi, phi_up)

        # Setup solver
        opt = builder.build()
        #self.solver = optas.CasADiSolver(opt).setup('ipopt')
        self.solver = optas.ScipyMinimizeSolver(opt).setup('SLSQP')

        # For later
        self.state_name = state_name
        self.Tmax = float(T-1)*dt
        self.T = T
        self.nX = nX
        self.state_alias_for_rectangle = state_alias_for_rectangle

    def plan(self, GpS0, GthetaS0, GpST, GthetaST):
        state_x_init = optas.DM.zeros(self.nX, self.T)
        for k in range(self.T):
            alpha = float(k)/float(self.T-1)
            state_x_init[:2,k] = optas.DM(GpS0) * (1-alpha) + alpha*optas.DM(GpST)
            state_x_init[2, k] = GthetaS0 * (1-alpha) + alpha * GthetaST
        
        self.solver.reset_initial_seed({
            'state/x': state_x_init,
            'control/u': 0.01*optas.DM.ones(4, self.T-1)
        })
        
        self.solver.reset_parameters({
            'GpS0': GpS0,
            'GthetaS0': GthetaS0,
            'GpST': GpST,
            'GthetaST': GthetaST,
        })
    
        solution = self.solver.solve()
        optas.np.set_printoptions(suppress=True, precision=3, linewidth=1000)
        slider_traj = solution['state/x']
        slider_traj_dx = solution['control/u']
        slider_plan = self.solver.interpolate(slider_traj, self.Tmax)
        slider_plan_dx = self.solver.interpolate(slider_traj_dx, self.Tmax)
        return slider_plan, slider_plan_dx
    
    def state_to_pose(self, state):
        SpC = optas.vertcat(0.5*self.Ly, 0.5*self.Ly*optas.tan(state[3]))
        GpC = state[:2] + rot2(state[2] + state[3] - 0.5*optas.np.pi)@SpC
        dr = rot2(state[2] + state[3] - 0.5*optas.np.pi) @ optas.vertcat(optas.cos(-0.5*optas.np.pi), optas.sin(-0.5*optas.np.pi))
        GpC -= dr*self.eff_ball_radius  # accounts for end effector ball radius
        endeff_position = GpC.toarray().flatten().tolist()
        box_position = state[:2].tolist() + [0.06]
        base_orientation=yaw2quat(state[2]).toarray().flatten()
        return endeff_position, box_position, base_orientation

class Animate:

    def __init__(self, animate):

        # Setup planner
        self.planner = Planner(0.1, 0.4, 0.2)
        # self.init = [-1.5, -1.5] # GpS0
        # self.goal = [1, 1]       # GpST
        self.endeffector_location_init = optas.np.array([0.4, 0.])
        GxS0, GyS0 = -1.5, -1.5
        GxST, GyST = 1, 1
        self.GpS0 = [GxS0, GyS0]
        self.GpST = [GxST, GyST]
        self.GthetaS0 = -0.*optas.np.pi    # GthetaS0
        self.GthetaST = 0.*optas.np.pi    # GthetaST
        self.rectangle_init = [self.GpS0[0] - self.planner.Lx, self.GpS0[1] - self.planner.Ly]
        self.rectangle_goal = [self.GpST[0] - self.planner.Lx, self.GpST[1] - self.planner.Ly]

        self.plan_x, self.plan_dx = self.planner.plan(self.GpS0, self.GthetaS0, self.GpST, self.GthetaST)

        # Setup figure
        self.t = optas.np.linspace(0, self.planner.duration, self.planner.T)
        self.X = self.plan_x(self.t)
        self.dX = self.plan_dx(self.t)

        self.fig, self.ax = plt.subplot_mosaic([['birdseye', 'position'],
                                                ['birdseye', 'velocity']],
                                               layout='constrained',
                                               figsize=(10, 5),
        )

        self.ax['birdseye'].plot(self.X[0, :], self.X[1, :], '-kx', label='plan')
        self.ax['birdseye'].add_patch(plt.Rectangle(self.rectangle_init, width=2*self.planner.Lx, height=2*self.planner.Ly, color='green', alpha=0.5))
        self.ax['birdseye'].add_patch(plt.Rectangle(self.rectangle_goal, width=2*self.planner.Lx, height=2*self.planner.Ly, color='red', alpha=0.5))
        self.obs_pos = optas.np.array(self.planner.obs_position)
        self.ax['birdseye'].add_patch(plt.Circle(self.obs_pos, radius=self.planner.obs_rad, color='black'))
        #self.ax['birdseye'].add_patch(plt.Circle(self.rectangle_goal, radius=self.planner.eff_ball_radius, color='blue', alpha=0.5))
        self.ax['birdseye'].set_aspect('equal')
        #print(self.planner.state_alias_for_rectangle.dlim)
        self.ax['birdseye'].set_xlim(*self.planner.state_alias_for_rectangle.dlim[0])
        self.ax['birdseye'].set_ylim(*self.planner.state_alias_for_rectangle.dlim[0])
        self.ax['birdseye'].set_title('Birdseye View')
        self.ax['birdseye'].set_xlabel('x')
        self.ax['birdseye'].set_ylabel('y')

        self.ax['position'].plot(self.t, self.X[0,:], '-rx', label='plan-x')
        self.ax['position'].plot(self.t, self.X[1,:], '-bx', label='plan-y')
        self.ax['position'].set_ylabel('Position')
        self.ax['position'].set_xlim(0, self.planner.duration)

        axlim = max([abs(l) for l in self.planner.state_alias_for_rectangle.dlim[0]])
        self.ax['position'].set_ylim(-axlim, axlim)

        self.ax['velocity'].plot(self.t, self.dX[0,:], '-rx', label='plan-dx')
        self.ax['velocity'].plot(self.t, self.dX[1,:], '-bx', label='plan-dy')
        self.ax['velocity'].axhline(self.planner.state_alias_for_rectangle.dlim[1][0], color='red', linestyle='--')
        self.ax['velocity'].axhline(self.planner.state_alias_for_rectangle.dlim[1][1], color='red', linestyle='--', label='limit')
        self.ax['velocity'].set_ylabel('Velocity')
        self.ax['velocity'].set_xlabel('Time')

        self.ax['velocity'].set_xlim(0, self.planner.duration)
        axlim = max([abs(1.5*l) for l in self.planner.state_alias_for_rectangle.dlim[1]])
        self.ax['velocity'].set_ylim(-axlim, axlim)

        for a in self.ax.values():
            a.legend(ncol=3, loc='lower right')
            a.grid()

        # Animate
        if not animate: return
        self.pos_line = self.ax['position'].axvline(color='blue', alpha=0.5)
        self.vel_line = self.ax['velocity'].axvline(color='blue', alpha=0.5)
        self.rectangle_visualisation = plt.Rectangle(xy=self.rectangle_init, width=2*self.planner.Lx, height=2*self.planner.Ly, angle=0.0, color='black')
        self.endeffector_visualisation = plt.Circle(self.endeffector_location_init, radius=self.planner.eff_ball_radius, color='blue')
        self.ani = FuncAnimation(self.fig, self.update, frames=self.t, blit=True)
        #animation save as video:
        #f = r"/home/sandorfelber/libraries/optas/example/animation_with_endeffector.mp4" 
        #writermp4 = animation.FFMpegWriter(fps=60) 
        #self.ani.save(f, writer=writermp4)

    def update(self, frame):

        # Update position/velocity indicator line
        self.pos_line.set_xdata([frame, frame])
        self.vel_line.set_xdata([frame, frame])

        # Update point mass
        state = self.plan_x(frame)
        #print(state)
        endeff_position, box_position, base_orientation = self.planner.state_to_pose(state)
        #print("self.plan_x(frame)", self.plan_x(frame))
        #print(endeff_position)
        self.rectangle_visualisation.set_xy(np.array([state[0], state[1]]) - np.array([self.planner.Lx, self.planner.Ly]))
        self.rectangle_visualisation.set_angle(yaw2deg(np.array([state[2]])))
        self.endeffector_visualisation.set_center(np.array([endeff_position[0], endeff_position[1]]))
        # self.rectangle_visualisation.set_xy([ (np.array(self.plan_x(frame)[1]) - np.array(self.planner.Lx)), np.array(self.plan_x(frame)[3]) - np.array([self.planner.Lx, self.planner.Ly]) ])
        self.ax['birdseye'].add_patch(self.rectangle_visualisation)
        self.ax['birdseye'].add_patch(self.endeffector_visualisation)

        return (self.rectangle_visualisation, self.endeffector_visualisation, self.pos_line, self.vel_line)

    @staticmethod
    def show():
        plt.show()

def main():
    from sys import argv
    animate = '--noanimate' not in argv
    Animate(animate).show()

if __name__ == '__main__':
    main()
