import os
import sys
import math
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PyBullet
import pybullet_api

# OpTaS
import optas
from optas.spatialmath import *

from example.zzz_tompccplanner import TOMPCCPlanner
from example.zzz_ik import IK

class PlannerVisualizer:
    def __init__(self, dt, Lx, Ly, GpS0, GthetaS0, GpST, GthetaST, planner, animate=True):
        self.planner = TOMPCCPlanner(dt, Lx, Ly)
        #to_mpcc_planner = TOMPCCPlanner(dt, Lx, Ly)
        # Setup planner
        self.init = [-1.5, -1.5]
        self.goal = [1, 1]
        plan_result = self.planner.plan(GpS0, GthetaS0, GpST, GthetaST)
        self.plan_x = plan_result["x"].toarray()
        self.plan_dx = plan_result["dx"].toarray()
        # self.plan_x, self.plan_dx = self.planner.plan(GpS0, GthetaS0, GpST, GthetaST)
        # to_mpcc_planner.plan(GpS0, GthetaS0, GpST, GthetaST)
        # Setup current state and controller
        self.curr = self.init
        self.dcurr = [0., 0.]
        self.controller = IK()

        # Setup figure
        self.t = np.linspace(0, dt, self.planner.T)
        self.X = self.plan_x(self.t)
        self.dX = self.plan_dx(self.t)

        self.fig, self.ax = plt.subplot_mosaic([['birdseye', 'position'],
                                                ['birdseye', 'velocity']],
                                               layout='constrained',
                                               figsize=(10, 5),
        )

        self.mpc_line, = self.ax['birdseye'].plot([], [], '-x', color='yellow', label='mpc')
        self.ax['birdseye'].plot(self.X[0, :], self.X[1, :], '-kx', label='plan')
        self.ax['birdseye'].add_patch(plt.Circle(self.init, radius=self.planner.pm_radius, color='green', alpha=0.5))
        self.ax['birdseye'].add_patch(plt.Circle(self.goal, radius=self.planner.pm_radius, color='red', alpha=0.5))
        self.dt = self.planner.dt
        self.obs_pos = np.array(self.planner.obs)
        self.obs_visual = plt.Circle(self.obs_pos, radius=self.planner.obs_rad, color='black')
        self.ax['birdseye'].add_patch(self.obs_visual)
        self.ax['birdseye'].set_aspect('equal')
        self.ax['birdseye'].set_xlim(*self.planner.point_mass.dlim[0])
        self.ax['birdseye'].set_ylim(*self.planner.point_mass.dlim[0])
        self.ax['birdseye'].set_title('Birdseye View')
        self.ax['birdseye'].set_xlabel('x')
        self.ax['birdseye'].set_ylabel('y')

        self.ax['position'].plot(self.t, self.X[0,:], '-rx', label='plan-x')
        self.ax['position'].plot(self.t, self.X[1,:], '-bx', label='plan-y')
        self.pm_pos_curr_x, = self.ax['position'].plot([], [], 'or', label='curr-x')
        self.pm_pos_curr_y, = self.ax['position'].plot([], [], 'ob', label='curr-y')
        self.ax['position'].set_ylabel('Position')
        self.ax['position'].set_xlim(0, self.planner.duration)

        axlim = max([abs(l) for l in self.planner.point_mass.dlim[0]])
        self.ax['position'].set_ylim(-axlim, axlim)

        self.ax['velocity'].plot(self.t, self.dX[0,:], '-rx', label='plan-dx')
        self.ax['velocity'].plot(self.t, self.dX[1,:], '-bx', label='plan-dy')
        self.pm_vel_curr_x, = self.ax['velocity'].plot([], [], 'or', label='curr-dx')
        self.pm_vel_curr_y, = self.ax['velocity'].plot([], [], 'ob', label='curr-dy')
        self.ax['velocity'].axhline(self.planner.point_mass.dlim[1][0], color='red', linestyle='--')
        self.ax['velocity'].axhline(self.planner.point_mass.dlim[1][1], color='red', linestyle='--', label='limit')
        self.ax['velocity'].set_ylabel('Velocity')
        self.ax['velocity'].set_xlabel('Time')

        self.ax['velocity'].set_xlim(0, self.planner.duration)
        axlim = max([abs(1.5*l) for l in self.planner.point_mass.dlim[1]])
        self.ax['velocity'].set_ylim(-axlim, axlim)

        for a in self.ax.values():
            a.legend(ncol=3, loc='lower right')
            a.grid()

    # Animate
        if not animate: return
        self.pos_line = self.ax['position'].axvline(color='blue', alpha=0.5)
        self.vel_line = self.ax['velocity'].axvline(color='blue', alpha=0.5)
        self.pm_visual = plt.Circle(self.init, radius=self.planner.pm_radius, color='blue', alpha=0.5)
        self.frames = list(range(self.planner.T))
        self.ani = FuncAnimation(self.fig, self.update, frames=self.frames, blit=True)

    def update(self, i):
        self.curr = self.planner.X[i,:]
        self.dcurr = self.planner.dX[i,:]
        x, y = self.curr
        dx, dy = self.dcurr
        self.pm_pos_curr_x.set_data(i*self.dt, x)
        self.pm_pos_curr_y.set_data(i*self.dt, y)
        self.pm_vel_curr_x.set_data(i*self.dt, dx)
        self.pm_vel_curr_y.set_data(i*self.dt, dy)
        self.pm_visual.center = self.curr
        self.obs_visual.center = self.obs_pos
        self.pos_line.set_xdata([i*self.dt, i*self.dt])
        self.vel_line.set_xdata([i*self.dt, i*self.dt])
        self.mpc_line.set_data(self.controller.ref[i])
        return self.pm_visual, self.obs_visual, self.pos_line, self.vel_line, self.pm_pos_curr_x, self.pm_pos_curr_y, self.pm_vel_curr_x, self.pm_vel_curr_y, self.mpc_line
