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

class IK:

    def __init__(self, dt, thresh_angle):

        cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory
        pi = optas.np.pi  # 3.141...
        T = 1 # no. time steps in trajectory
        link_ee = 'end_effector_ball'  # end-effector link name

        # Setup robot
        urdf_filename = os.path.join(cwd, 'robots', 'kuka_lwr', 'kuka_lwr.urdf')
        kuka = optas.RobotModel(
            urdf_filename=urdf_filename,
            time_derivs=[1],  # i.e. joint velocity
        )
        kuka_name = kuka.get_name()

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=T, robots=[kuka], derivs_align=True)

        # Setup parameters
        qc = builder.add_parameter('qc', kuka.ndof)  # current robot joint configuration
        pg = builder.add_parameter('pg', 3)  # goal end-effector position

        # Get joint velocity
        dq = builder.get_model_state(kuka_name, t=0, time_deriv=1)

        # Get next joint state
        q = qc + dt*dq

        # Get jacobian
        Jl = kuka.get_global_linear_jacobian(link_ee, qc)

        # Get end-effector velocity
        dp = Jl @ dq

        # Get current end-effector position
        pc = kuka.get_global_link_position(link_ee, qc)

        # Get next end-effector position
        p = pc + dt*dp

        # Cost: match end-effector position
        diffp = p - pg
        W_p = optas.diag([1e2, 1e2, 1e3])
        builder.add_cost_term('match_p', diffp.T @ W_p @ diffp)

        # Cost: min joint velocity
        w_dq = 0.01
        builder.add_cost_term('min_dq', w_dq*optas.sumsqr(dq))

        # Get global z-axis of end-effector
        T = kuka.get_global_link_transform(link_ee, q)
        z = T[:3, 2]

        # Constraint: eff orientation
        e = optas.DM([0, 0, -1.])
        builder.add_leq_inequality_constraint('eff_orien', optas.cos(thresh_angle), e.T @ z)

        # Cost: align eff
        w_ori = 1e4
        builder.add_cost_term('eff_orien', w_ori*optas.sumsqr(e.T @ z - 1))

        # Setup solver
        optimization = builder.build()
        self.solver = optas.CasADiSolver(optimization).setup('sqpmethod')

        # Setup variables required later
        self.kuka_name = kuka_name

    def compute_target_velocity(self, qc, pg):
        self.solver.reset_parameters({'qc': optas.DM(qc), 'pg': optas.DM(pg)})
        solution = self.solver.solve()
        return solution[f'{self.kuka_name}/dq'].toarray().flatten()