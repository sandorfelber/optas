# Python standard lib
import os
import sys
import math
import pathlib

# PyBullet
import pybullet_api

# OpTaS
import optas
from optas.spatialmath import *


class TOMPCCPlanner:

    def __init__(self, dt, Lx, Ly):

        # Setup
        mu = 0.1  # coef of friction
        dt = float(dt)  # time step
        nX = 4  # number of state variables
        nU = 4  # number of control variables
        T = 20  # number of step
        Lx = float(Lx)  # length of slider (box) in x-axis
        Ly = float(Ly)  # length of slider (box) in y-axis
        phi_lo = math.atan2(Lx, Ly) # lower limit for phi
        phi_up = -phi_lo # upper limit for phi
        L = optas.diag([1, 1, 0.5])  # limit surface model
        SxC0 = 0.5*Ly # initial contact position in x-axis of slider
        SyC0 = 0. # initial contact position in y-axis of slider
        SphiC0 = 0. # initial contact orientation in slider frame
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

        # Setup task models
        state = optas.TaskModel('state', dim=nX, time_derivs=[0])
        control = optas.TaskModel('control', dim=nU, time_derivs=[0], T=T-1, symbol='u')

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
        # self.solver = optas.CasADiSolver(opt).setup('ipopt')
        self.solver = optas.ScipyMinimizeSolver(opt).setup('SLSQP')

        # For later
        self.Tmax = float(T-1)*dt
        self.T = T
        self.nX = nX

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
        print("Solution:", solution)
        optas.np.set_printoptions(suppress=True, precision=3, linewidth=1000)
        slider_traj = solution['state/x']
        print("Slider_traj", type(slider_traj), slider_traj)
        slider_plan = self.solver.interpolate(slider_traj, self.Tmax)
        print("Slider Plan", slider_plan)
        return slider_traj