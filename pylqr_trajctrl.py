"""
LQR based trajectory controller
"""
import numpy as np

import pylqr

class PyLQR_TrajCtrl():
    """
    Use second-order system and acceleration as system and input
    The trajectory is given as a set of reference waypoints and associated tracking weights
    """
    def __init__(self, R=.01, dt=0.01):
        #control penalty, smoothness of the trajectory
        self.R_ = R
        self.dt_ = dt
        self.Q_vel_ratio_ = 10

        #desired functions for plant dynamics and cost
        self.plant_dyn_ = None
        self.plant_dyn_dx_ = None
        self.plant_dyn_du_ = None

        self.cost_ = None
        self.cost_dx_ = None
        self.cost_du_ = None
        self.cost_dxx_ = None
        self.cost_duu_ = None
        self.cost_dux_ = None

        self.ilqr_ = None
        return

    def build_ilqr_solver(self, ref_pnts, weight_mats):
        #figure out dimension
        self.T_ = len(ref_pnts)
        self.n_dims_ = len(ref_pnts[0])

        self.ref_array = np.copy(ref_pnts)
        self.weight_array = [mat for mat in weight_mats]
        #clone weight mats if there are not enough weight mats
        for i in range(self.T_ - len(self.weight_array)):
            self.weight_array.append(self.weight_array[-1])

        #build dynamics, second-order linear dynamical system
        self.A_ = np.eye(self.n_dims_*2)
        self.A_[0:self.n_dims_, self.n_dims_:] = np.eye(self.n_dims_) * self.dt_
        self.B_ = np.zeros((self.n_dims_*2, self.n_dims_))
        self.B_[self.n_dims_:, :] = np.eye(self.n_dims_) * self.dt_

        self.plant_dyn_ = lambda x, u, t, aux: self.A_.dot(x) + self.B_.dot(u)
        self.plant_dyn_dx_ = lambda x, u, t, aux: self.A_
        self.plant_dyn_du_ = lambda x, u, t, aux: self.B_

        #build cost functions, quadratic ones
        def tmp_cost_func(x, u, t, aux):
            err = x[0:self.n_dims_] - self.ref_array[t]
            cost = (err).dot(self.weight_array[t]).dot(err) + np.linalg.norm(u)**2 * self.R_
            if t > self.T_-1:
                #regularize velocity for the termination point
                cost += np.linalg.norm(x[self.n_dims_:])**2  * self.R_ * self.Q_vel_ratio_
            return cost
        
        self.cost_ = tmp_cost_func

        def tmp_cost_func_dx(x, u, t, aux):
            err = x[0:self.n_dims_] - self.ref_array[t]
            grad = np.concatenate([2*err.dot(self.weight_array[t]), np.zeros(self.n_dims_)])
            if t > self.T_-1:
                grad[self.n_dims_:] = grad[self.n_dims_:] + 2 * self.R_ * self.Q_vel_ratio_ * x[self.n_dims_, :]
            return grad

        self.cost_dx_ = tmp_cost_func_dx

        self.cost_du_ = lambda x, u, t, aux: 2 * self.R_ * u

        def tmp_cost_func_dxx(x, u, t, aux):
            hessian = np.zeros((2*self.n_dims_, 2*self.n_dims_))
            hessian[0:self.n_dims_, 0:self.n_dims_] = 2 * self.weight_array[t]

            if t > self.T_-1:
                hessian[self.n_dims_:, self.n_dims_:] = 2 * np.eye(self.n_dims_) * self.R_ * self.Q_vel_ratio_
            return hessian

        self.cost_dxx_ = tmp_cost_func_dxx

        self.cost_duu_ = lambda x, u, t, aux: 2 * self.R_ * np.eye(self.n_dims_)
        self.cost_dux_ = lambda x, u, t, aux: np.zeros((self.n_dims_, 2*self.n_dims_))

        #build an iLQR solver based on given functions...
        self.ilqr_ = pylqr.PyLQR_iLQRSolver(T=self.T_-1, plant_dyn=self.plant_dyn_, cost=self.cost_)
        self.ilqr_.plant_dyn_dx = self.plant_dyn_dx_
        self.ilqr_.plant_dyn_du = self.plant_dyn_du_
        self.ilqr_.cost_dx = self.cost_dx_
        self.ilqr_.cost_du = self.cost_du_
        self.ilqr_.cost_dxx = self.cost_dxx_
        self.ilqr_.cost_duu = self.cost_duu_
        self.ilqr_.cost_dux = self.cost_dux_

        return

    def synthesize_trajectory(self, x0):
        if self.ilqr_ is None:
            print 'No iLQR solver has been prepared.'
            return None

        #initialization doesn't matter as global optimality can be guaranteed?
        u_init = [np.zeros(self.n_dims_) for i in range(self.T_-1)]
        x_init = np.concatenate([x0, np.zeros(self.n_dims_)])
        res = self.ilqr_.ilqr_iterate(x_init, u_init, n_itrs=50, tol=1e-6, verbose=True)
        return res['x_array_opt'][:, 0:self.n_dims_]

"""
Test case, 2D trajectory to track a sinuoidal..
"""
import matplotlib.pyplot as plt

def PyLQR_TrajCtrl_Test():
    n_pnts = 200
    x_coord = np.linspace(0.0, 2*np.pi, n_pnts)
    y_coord = np.sin(x_coord)
    #concatenate to have trajectory
    ref_traj = np.array([x_coord, y_coord]).T
    weight_mats = [ np.eye(ref_traj.shape[1])*100 ]

    #draw reference trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], '.-k', linewidth=3.5)
    ax.plot([ref_traj[0, 0]], [ref_traj[0, 1]], '*k', markersize=16)

    lqr_traj_ctrl = PyLQR_TrajCtrl()
    lqr_traj_ctrl.build_ilqr_solver(ref_traj, weight_mats)

    n_queries = 5

    for i in range(n_queries):
        #start from a perturbed point
        x0 = ref_traj[0, :] + np.random.rand(2) * 2 - 1
        syn_traj = lqr_traj_ctrl.synthesize_trajectory(x0)
        #plot it
        ax.plot(syn_traj[:, 0], syn_traj[:, 1], linewidth=3.5)

    plt.show()
    return

if __name__ == '__main__':
    PyLQR_TrajCtrl_Test()