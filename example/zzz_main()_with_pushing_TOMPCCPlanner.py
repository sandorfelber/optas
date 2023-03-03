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

import numpy as np
import pybullet_api
from tompccplanner import TOMPCCPlanner
from planner_visualizer import PlannerVisualizer
from ik import IK

def yaw2quat(angle):
    return Quaternion.fromrpy(tr2eul(rotz(angle))).getquat()

# Setup PyBullet
qc = -np.deg2rad([0, 30, 0, -90, 0, 60, 0])
q = qc.copy()
eff_ball_radius = 0.015
hz = 50
dt = 1.0/float(hz)
pb = pybullet_api.PyBullet(
    dt,
    camera_distance=0.5,
    camera_target_position=[0.3, 0.2, 0.],
    camera_yaw=135,
)
kuka = pybullet_api.KukaLWR()
kuka.reset(qc)
GxS0 = 0.4
GyS0 = 0.065
GthetaS0 = 0.
box_base_position = [GxS0, GyS0, 0.06]
Lx = 0.2
Ly = 0.1
box_half_extents = [0.5*Lx, 0.5*Ly, 0.06]
box = pybullet_api.DynamicBox(
    base_position=box_base_position,
    half_extents=box_half_extents,
)
GxST = 0.45
GyST = 0.3
GthetaST = 0.1*np.pi
pybullet_api.VisualBox(
    base_position=[GxST, GyST, 0.06],
    base_orientation=yaw2quat(GthetaST).toarray().flatten(),
    half_extents=box_half_extents,
    rgba_color=[1., 0., 0., 0.5],
)
plan_box = pybullet_api.VisualBox(
    base_position=[GxS0, GyS0, 0.06],
    half_extents=box_half_extents,
    rgba_color=[0., 0., 1., 0.5],
)

# Setup TO MPCC planner
to_mpcc_planner = TOMPCCPlanner(0.1, Lx, Ly)

# Setup IK
thresh_angle = np.deg2rad(30.)
ik = IK(dt, thresh_angle)

# Start pybullet
pb.start()
start_time = pybullet_api.time.time()

# Move robot to start position
Tmax_start = 6.
pginit = np.array([0.4, 0., 0.06])
while True:
    t = pybullet_api.time.time() - start_time
    if t > Tmax_start:
        break
    dqgoal = ik.compute_target_velocity(q, pginit)
    q += dt*dqgoal
    kuka.cmd(q)
    pybullet_api.time.sleep(dt)

# Plan a trajectory
GpS0 = [GxS0, GyS0]
GpST = [GxST, GyST]
plan = to_mpcc_planner.plan(GpS0, GthetaS0, GpST, GthetaST)
sim = PlannerVisualizer(dt, Lx, Ly, GpS0, GthetaS0, GpST, GthetaST, TOMPCCPlanner, animate=True)

# Main loop
p = pginit.copy()
start_time = pybullet_api.time.time()
while True:
    t = pybullet_api.time.time() - start_time
    if t > to_mpcc_planner.Tmax:
        break
    boxpose = box.get_pose()
    dqgoal = ik.compute_target_velocity(q, p)
    q += dt*dqgoal
    state = plan(t)
    SpC = optas.vertcat(0.5*Ly, 0.5*Ly*optas.tan(state[3]))
    GpC = state[:2] + rot2(state[2] + state[3] - 0.5*optas.np.pi)@SpC
    dr = rot2(state[2] + state[3] - 0.5*optas.np.pi) @ optas.vertcat(optas.cos(-0.5*optas.np.pi), optas.sin(-0.5*optas.np.pi))
    GpC -= dr*eff_ball_radius  # accounts for end effector ball radius
    p = GpC.toarray().flatten().tolist() + [0.06]
    box_position = state[:2].tolist() + [0.06]
    plan_box.reset(
        base_position=box_position,
        base_orientation=yaw2quat(state[2]).toarray().flatten(),
    )
    kuka.cmd(q)
    pybullet_api.time.sleep(dt)
