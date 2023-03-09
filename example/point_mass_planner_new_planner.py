import optas
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from optas.spatialmath import *
import math
import numpy as np

def yaw2quat(angle):
    return Quaternion.fromrpy(tr2eul(rotz(angle))).getquat()

def quatsplit(quat):
    return Quaternion.split(quat)

def yaw2deg(angle):
    return angle*180/np.pi

def quat2deg(q):
    """Converts a quaternion to Euler angles in degrees."""
    q = np.array(q, dtype=np.float64)
    pitch = np.arcsin(2 * (q[0]*q[2] - q[3]*q[1]))
    roll = np.arctan2(2 * (q[0]*q[1] + q[2]*q[3]), 1 - 2 * (q[1]**2 + q[2]**2))
    yaw = np.arctan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
    return np.array([np.degrees(pitch), np.degrees(roll), np.degrees(yaw)])


# TOMPCCPlanner
class Planner:

    def __init__(self, dt, Lx, Ly):

        # Setup
        mu = 0.1  # coef of friction
        dt = float(dt)  # time step
        self.dt = dt
        nX = 4  # number of state variables
        nU = 4  # number of control variables
        T = 50  # number of step
        Lx = float(Lx)  # length of slider (box) in x-axis
        Ly = float(Ly)  # length of slider (box) in y-axis
        eff_ball_radius = 0.015
        self.Lx = Lx
        self.Ly = Ly
        self.eff_ball_radius = eff_ball_radius
        endeffector_location_init = optas.np.array([0.4, 0., 0.06])
        #obs_position = [0, 0]  # obstacle position
        #obs_rad = 0.25  # radius of obstacle
        #self.obs_position = obs_position
        #self.obs_rad = obs_rad
        phi_lo = math.atan2(Lx, Ly) # lower limit for phi
        phi_up = -phi_lo # upper limit for phi
        L = optas.diag([1, 1, 0.5])  # limit surface model
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
        dlim = {0: [-0.25, 1], 1: [-0.1, 1]}  # pos/vel limits

        # Setup task models
        state = optas.TaskModel('state', dim=nX, time_derivs=[0], dlim=dlim)
        state = state
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

        # Constraint: limits
        #builder.enforce_model_limits(state_name, time_deriv=0)
        #builder.enforce_model_limits(state_name, time_deriv=1)

        # Get states/controls
        X = builder.get_model_states('state')
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXX")
        #print(X)
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXX")
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
        f_list = []
        for k in range(T-1):

            # Setup
            xn = X[:, k+1]  # next state
            x = X[:, k]   # current state
            u = U[:, k]  # control input
            R = rotz(theta[k]-0.5*optas.np.pi)  # rotation matrix in xy-plane of slider
            SxC = 0.5*Ly # x-position of pusher wrt center of box
            SyC = SxC*optas.tan(phi[0]) # y-position of contact
            JC = optas.horzcat(I, optas.vertcat(-SyC, SxC))

            # Compute system dynamics f(x, u) = Ku
            K = optas.vertcat(
                optas.horzcat(R @ L @ JC.T, o3, o3),
                optas.horzcat(o2.T, 1, -1),
            )
            f = K @ u
            
            #f_list.append(f)
            #print(f_list)
            #f_array = np.array(f_list)
            # Add constraint
            builder.add_equality_constraint(f'dynamics_{k}', xn, x + dt*f)
        #f_list = f_list
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
        '''    
        # Constraint: obstacle avoidance
        #X = builder.get_model_states(rectangle_task_model_name)
        rectangle_dist_center2corner = (Lx**2 + Ly**2)**0.5
        #print("RECTANGLE CONRER TO CENTER", rectangle_dist_center2corner)
        safe_dist_sq = (obs_rad + rectangle_dist_center2corner)**2
        for i in range(T):
            #print("X[:, i]", X[2:, i])
            dist_sq1 = optas.sumsqr(obs_position - X[:2, i])  # first two elements of X are position and the second two are velocity??
            dist_sq2 = optas.sumsqr(obs_position - X[2:, i])  # first two elements of X are position and the second two are velocity??
            #print("dist_sq1", dist_sq1)
            #print("dist_sq2", dist_sq2)
            builder.add_geq_inequality_constraint(f'obs_avoid1_{i}', dist_sq1, safe_dist_sq)
            builder.add_geq_inequality_constraint(f'obs_avoid2_{i}', dist_sq2, safe_dist_sq)
        ''' 
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
        print("SQLSP called")
        # For later
        self.state_name = state_name
        self.T = T
        self.nX = nX
        self.state = state


    def plan(self, GpS0, GthetaS0, GpST, GthetaST):
        state_x_init = optas.DM.zeros(self.nX, self.T)
        for k in range(self.T):
            alpha = float(k)/float(self.T-1)
            state_x_init[:2,k] = optas.DM(GpS0) * (1-alpha) + alpha*optas.DM(GpST)
            state_x_init[2, k] = GthetaS0 * (1-alpha) + alpha * GthetaST
            #print(state_x_init)
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
        #print(solution)
        optas.np.set_printoptions(suppress=True, precision=3, linewidth=1000)
        slider_traj = solution['state/x']
        #print(slider_traj)
        #slider_traj_dx = solution['state/x'] * self.dt                   #'control/u']
        slider_plan = self.solver.interpolate(slider_traj, self.duration)
        #slider_plan_dx = self.solver.interpolate(slider_traj_dx, self.duration)
        return slider_plan#, slider_plan_dx
    
    def state_to_pose(self, state):
        SpC = optas.vertcat(0.5*self.Ly, 0.5*self.Ly*optas.tan(state[3]))
        #print(state[3])
        GpC = state[:2] + rot2(state[2] + state[3] - 0.5*optas.np.pi)@SpC
        dr = rot2(state[2] + state[3] - 0.5*optas.np.pi) @ optas.vertcat(optas.cos(-0.5*optas.np.pi), optas.sin(-0.5*optas.np.pi))
        #print(self.eff_ball_radius)
        GpC -= dr*self.eff_ball_radius  # accounts for end effector ball radius
        #print(GpC)
        endeff_position = [GpC.toarray().flatten().tolist()[0] - self.eff_ball_radius, GpC.toarray().flatten().tolist()[1] - self.eff_ball_radius]
        box_position = state[:2].tolist() #+ [0.06]
        #base_orientation=[yaw2quat(state[2]).toarray().flatten()[2], yaw2quat(state[2]).toarray().flatten()[3]]
        base_orientation_rotation = R.from_quat([yaw2quat(state[2]).toarray().flatten()[0], yaw2quat(state[2]).toarray().flatten()[1], yaw2quat(state[2]).toarray().flatten()[2], yaw2quat(state[2]).toarray().flatten()[3]])
        base_orientation = base_orientation_rotation.as_euler('zyx', degrees=True)[0] #base_orientation in degrees (about z axis)
        #print("box:", box_position)
        #print("endeff:",endeff_position)
        return endeff_position, box_position, base_orientation

class Animate:

    def __init__(self, animate):

        # Setup planner
        hz = 50
        dt = 1.0/float(hz)
        self.planner = Planner(0.1, Lx = 0.2, Ly = 0.1)
        # self.init = [-1.5, -1.5] # GpS0
        # self.goal = [1, 1]       # GpST
        self.endeffector_location_init = optas.np.array([0.4, 0.]) #pginit
        GxS0, GyS0 = 0.0, 0.0
        GxST, GyST = 0.3, 0.6 # 0.45, 0.3
        self.GthetaS0 = 0.0*optas.np.pi    # GthetaS0
        self.GthetaST = 0.1*optas.np.pi    # GthetaST
        self.rectangle_position_init = [GxS0 - 0.5 * self.planner.Lx, GyS0 - 0.5 * self.planner.Ly]
        self.rectangle_position_goal = [GxST - 0.5 * self.planner.Lx, GyST - 0.5 * self.planner.Ly]

        self.plan_x = self.planner.plan([GxS0, GyS0], self.GthetaS0, [GxST, GyST], self.GthetaST)
        #self.plan_x, self.plan_dx = self.planner.plan([GxS0, GyS0], self.GthetaS0, [GxST, GyST], self.GthetaST)

        # Setup figure
        self.t = optas.np.linspace(0, self.planner.duration, self.planner.T)
        self.X = self.plan_x(self.t)
        #print(self.X)
        self.dX = self.plan_x(self.t)
        self.dX[:,0] = 0  #starting velocity 0
        # velocity in x direction:
        self.dX[3] = np.roll(self.dX[0], 1)
        self.dX[3][0] = 0
        self.dX[0] = self.dX[0] - self.dX[3]
        self.dX[0][1] = self.dX[0][1] - GxS0
        print("self.dX1")
        print(self.dX)
        # velocity in y direction: 
        self.dX[3] = np.roll(self.dX[1], 1)
        self.dX[3][0] = 0
        self.dX[1] = self.dX[1] - self.dX[3]
        self.dX[1][1] = self.dX[1][1] - GyS0
        #print("self.dX2")
        #print(self.dX)
        # angular velocity along z axis: 
        self.dX[3] = np.roll(self.dX[2], 1)
        self.dX[3][0] = 0
        self.dX[2] = self.dX[2] - self.dX[3]
        self.dX[2][1] = self.dX[2][1] - self.GthetaS0
        #print("self.dX3")
        #print(self.dX)
        
        #print(self.dX)
        #self.endeff_trajectory = self.planner.state_to_pose(self.update.state)
        self.fig, self.ax = plt.subplot_mosaic([['birdseye', 'position'],
                                                ['birdseye', 'velocity']],
                                               layout='constrained',
                                               figsize=(10, 5),
        )

        self.ax['birdseye'].plot(self.X[0, :], self.X[1, :], '-kx', label='box_center_plan')
        #self.ax['birdseye'].plot(self.X[2, :], self.X[3, :], '-kx', label='pusher_plan')
        self.ax['birdseye'].add_patch(plt.Rectangle(self.rectangle_position_init, width=self.planner.Lx, height=self.planner.Ly, angle=yaw2deg(self.GthetaS0), color='green', alpha=0.5, rotation_point='center'))
        self.ax['birdseye'].add_patch(plt.Rectangle(self.rectangle_position_goal, width=self.planner.Lx, height=self.planner.Ly, angle=yaw2deg(self.GthetaST), color='red', alpha=0.5, rotation_point='center'))
        #self.obs_pos = optas.np.array(self.planner.obs_position)
        #self.ax['birdseye'].add_patch(plt.Circle(self.obs_pos, radius=self.planner.obs_rad, color='black'))
        #self.ax['birdseye'].add_patch(plt.Circle(self.rectangle_goal, radius=self.planner.eff_ball_radius, color='blue', alpha=0.5))
        self.ax['birdseye'].set_aspect('equal')
        #print(self.planner.state.dlim)
        self.ax['birdseye'].set_xlim(*self.planner.state.dlim[0])
        self.ax['birdseye'].set_ylim(*self.planner.state.dlim[0])
        self.ax['birdseye'].set_title('Birdseye View')
        self.ax['birdseye'].set_xlabel('x')
        self.ax['birdseye'].set_ylabel('y')
        #print(self.X)
        self.ax['position'].plot(self.t, self.X[0,:], '-rx', label='plan-x')
        self.ax['position'].plot(self.t, self.X[1,:], '-bx', label='plan-y')
        self.ax['position'].set_ylabel('Position')
        self.ax['position'].set_xlim(0, self.planner.duration)

        axlim = max([abs(l) for l in self.planner.state.dlim[0]])
        self.ax['position'].set_ylim(-axlim, axlim)

        self.ax['velocity'].plot(self.t, self.dX[0,:], '-rx', label='plan-dx')
        self.ax['velocity'].plot(self.t, self.dX[1,:], '-bx', label='plan-dy')
        self.ax['velocity'].axhline(self.planner.state.dlim[1][0], color='red', linestyle='--')
        self.ax['velocity'].axhline(self.planner.state.dlim[1][1], color='red', linestyle='--', label='limit')
        self.ax['velocity'].set_ylabel('Velocity')
        self.ax['velocity'].set_xlabel('Time')

        self.ax['velocity'].set_xlim(0, self.planner.duration)
        axlim = max([abs(1.5*l) for l in self.planner.state.dlim[1]])
        self.ax['velocity'].set_ylim(-axlim, axlim)

        for a in self.ax.values():
            a.legend(ncol=3, loc='lower right')
            a.grid()

        # Animate
        if not animate: return
        self.pos_line = self.ax['position'].axvline(color='blue', alpha=0.5)
        self.vel_line = self.ax['velocity'].axvline(color='blue', alpha=0.5)
        self.rectangle_visualisation = plt.Rectangle(xy=self.rectangle_position_init, width=self.planner.Lx, height=self.planner.Ly, angle=yaw2deg(self.GthetaS0), color='black', rotation_point = 'center')
        self.endeffector_visualisation = plt.Circle(self.endeffector_location_init, radius=self.planner.eff_ball_radius, color='blue')
        self.ani = FuncAnimation(self.fig, self.update, frames=self.t, blit=True)
        #animation save as video:
        f = r"/home/sandorfelber/Desktop/videos/Pusher_Slider_Animation_03_09_Diagonal_Path_2.mp4" 
        writermp4 = animation.FFMpegWriter(fps=10) 
        self.ani.save(f, writer=writermp4)

    def update(self, frame):

        # Update position/velocity indicator line
        self.pos_line.set_xdata([frame, frame])
        self.vel_line.set_xdata([frame, frame])
        
        # Update point mass
        state = self.plan_x(frame)
        self.state = state
        #print(state)
        endeff_position, box_position, base_orientation = self.planner.state_to_pose(state)
        #print(np.array(base_orientation))
        #print("self.plan_x(frame)", self.plan_x(frame))
        #print(endeff_position)
        self.rectangle_visualisation.set_xy(np.array([box_position[0], box_position[1]]) - np.array([self.planner.Lx*0.5, self.planner.Ly*0.5]))
        #self.rectangle_visualisation.rotation_point = center
        self.rectangle_visualisation.set_angle(base_orientation)
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
