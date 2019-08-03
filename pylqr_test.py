"""
A Inverted Pendulum test for the iLQR implementation
"""
from __future__ import print_function

try:
    import autograd.numpy as np
except ImportError:
    import numpy as np

import matplotlib.pyplot as plt

import pylqr

class InversePendulumProblem():
    def __init__(self):
        #parameters
        self.m_ = 1
        self.l_ = .5
        self.b_ = .1
        self.lc_ = .5
        self.g_ = 9.81
        self.dt_ = 0.01

        self.I_ = self.m_ * self.l_**2

        self.ilqr = None
        self.res = None
        self.T = 150

        self.Q = 100
        self.R = .01

        #terminal Q to regularize final speed
        self.Q_T = .1
        return

    def plant_dyn(self, x, u, t, aux):
        xdd = (u[0] - self.m_*self.g_*self.lc_*np.sin(x[0]) - self.b_*x[1])/self.I_

        # dont need term +0.5*(qdd**2)*self.dt_?
        x_new =  x + self.dt_ * np.array([x[1], xdd])
        return x_new
    
    def plant_dyn_dx(self, x, u, t, aux):
        dfdx = np.array([
            [1, self.dt_],
            [-self.m_*self.g_*self.lc_*np.cos(x[0])*self.dt_/self.I_, 1-self.b_*self.dt_/self.I_]
            ])
        return dfdx

    def plant_dyn_du(self, x, u, t, aux):
        dfdu = np.array([
            [0],
            [self.dt_/self.I_]
            ])
        return dfdu

    def cost_dx(self, x, u, t, aux):
        if t < self.T:
            dldx = np.array(
                [2*(x[0]-np.pi)*self.Q, 0]
                ).T
        else:
            #terminal cost
            dldx = np.array(
                [2*(x[0]-np.pi)*self.Q, 2*x[1]*self.Q_T]
                ).T
        return dldx

    def cost_du(self, x, u, t, aux):
        dldu = np.array(
            [2*u[0]*self.R]
            )
        return dldu

    def cost_dxx(self, x, u, t, aux):
        if t < self.T:
            dldxx = np.array([
                [2, 0],
                [0, 0]
                ]) * self.Q
        else:
            dldxx = np.array([
                [2* self.Q, 0],
                [0, 2*x[1]*self.Q_T]
                ]) 
        return dldxx
    def cost_duu(self, x, u, t, aux):
        dlduu = np.array([
            [2*self.R]
            ])
        return dlduu
    def cost_dux(self, x, u, t, aux):
        dldux = np.array(
            [0, 0]
            )
        return dldux

    def instaneous_cost(self, x, u, t, aux):
        if t < self.T:
            return (x[0] - np.pi)**2 * self.Q + u[0]**2 * self.R
        else:
            return (x[0] - np.pi)**2 * self.Q + x[1]**2*self.Q_T + u[0]**2 * self.R

    #grad_types = ['user', 'autograd', 'fd']
    def build_ilqr_problem(self, grad_type=0):
        if grad_type==0:
            self.ilqr = pylqr.PyLQR_iLQRSolver(T=self.T, plant_dyn=self.plant_dyn, cost=self.instaneous_cost, use_autograd=False)
            #not use finite difference, assign the gradient functions
            self.ilqr.plant_dyn_dx = self.plant_dyn_dx
            self.ilqr.plant_dyn_du = self.plant_dyn_du
            self.ilqr.cost_dx = self.cost_dx
            self.ilqr.cost_du = self.cost_du
            self.ilqr.cost_dxx = self.cost_dxx
            self.ilqr.cost_duu = self.cost_duu
            self.ilqr.cost_dux = self.cost_dux
        elif grad_type==1:
            self.ilqr = pylqr.PyLQR_iLQRSolver(T=self.T, plant_dyn=self.plant_dyn, cost=self.instaneous_cost, use_autograd=True)
        else:
            #finite difference
            self.ilqr = pylqr.PyLQR_iLQRSolver(T=self.T, plant_dyn=self.plant_dyn, cost=self.instaneous_cost, use_autograd=False)
        return

    def solve_ilqr_problem(self, x0=None):
        #prepare initial guess
        u_init = np.array([np.array([0]) for t in range(self.T)])
        if x0 is None:
            x0 = np.array([np.random.rand()*2*np.pi - np.pi, 0])

        if self.ilqr is not None:
            self.res = self.ilqr.ilqr_iterate(x0, u_init, n_itrs=150, tol=1e-6)
        return

    def plot_ilqr_result(self):
        if self.res is not None:
            #draw cost evolution and phase chart
            fig = plt.figure(figsize=(16, 8), dpi=80)
            ax_cost = fig.add_subplot(121)
            n_itrs = len(self.res['J_hist'])
            ax_cost.plot(np.arange(n_itrs), self.res['J_hist'], 'r', linewidth=3.5)
            ax_cost.set_xlabel('Number of Iterations', fontsize=20)
            ax_cost.set_ylabel('Trajectory Cost')

            ax_phase = fig.add_subplot(122)
            theta = self.res['x_array_opt'][:, 0]
            theta_dot = self.res['x_array_opt'][:, 1]
            ax_phase.plot(theta, theta_dot, 'k', linewidth=3.5)
            ax_phase.set_xlabel('theta (rad)', fontsize=20)
            ax_phase.set_ylabel('theta_dot (rad/s)', fontsize=20)
            ax_phase.set_title('Phase Plot', fontsize=20)

            ax_phase.plot([theta[-1]], [theta_dot[-1]], 'b*', markersize=16)

            plt.show()

        return



try:
    import torch
    from torch.utils.data import Dataset, DataLoader

except ImportError:
    print('Error in importing PyTorch package.')

class SimpleRNNModel(torch.nn.Module):
    def __init__(self, n_rnncell):
        super(SimpleRNNModel, self).__init__()
        self.lstm = torch.nn.LSTM(n_rnncell, n_rnncell)
        self.in_feat = torch.nn.Sequential(torch.nn.Linear(3, n_rnncell), torch.nn.ReLU())
        self.out_feat = torch.nn.Linear(n_rnncell, 2)
        self.n_rnncell = n_rnncell
    
    def step(self, x, u, hidden):
        output, hidden = self.lstm(self.in_feat(torch.cat((x, u), dim=1))[:, None, :], hidden)
        return self.out_feat(output), hidden
    
    def forward(self, x_seq, u_seq, hidden=None):
        #batch x len x dim
        T = u_seq.size()[1]
        x_new_seq = []
        hidden_new_seq = []

        for i in range(T):
            x_new, hidden_new = self.step(x_seq[:, i, :], u_seq[:, i, :], hidden)
            x_new_seq.append(x_new)
            hidden_new_seq.append(hidden_new[0])    #only store hidden state, i think cell state does not carry temporal information, right?
        
        #combine seq and remember to swap the axes
        x_new_seq = torch.cat(x_new_seq, dim=1)
        hidden_new_seq = torch.cat(hidden_new_seq, dim=1)
        return x_new_seq, hidden_new_seq

class SimpleFNNModel(torch.nn.Module):
    def __init__(self, n_latent):
        super(SimpleFNNModel, self).__init__()

        self.n_latent = n_latent

        self.in_feat = torch.nn.Sequential(
            torch.nn.Linear(3, n_latent),
            torch.nn.ReLU())
        self.out_feat = torch.nn.Linear(n_latent, 2)
    
    def step(self, x, u):
        #print(self.in_feat(torch.cat((x, u), dim=1)).shape)
        return self.out_feat(self.in_feat(torch.cat((x, u), dim=1)))
    
    def forward(self, x_seq, u_seq):
        #batch x len x dim
        T = u_seq.size()[1]
        x_new_seq = []

        for i in range(T):
            x_new = self.step(x_seq[:, i, :], u_seq[:, i, :])
            
            x_new_seq.append(x_new[:, None, :])
        
        #combine seq and remember to swap the axes
        x_new_seq = torch.cat(x_new_seq, dim=1)
        #print(x_new_seq.shape)
        return x_new_seq
    
class InvertedPendulumDataset(Dataset):
    def __init__(self, x_traj, u_traj):
        self.x_data = x_traj[:, :-1, :]
        self.u_data = u_traj
        self.x_new_data = x_traj[:, 1:, :]
    
    def __len__(self):
        return self.x_data.shape[0]
    def __getitem__(self, idx):
        return self.x_data[idx, :, :], self.u_data[idx, :, :], self.x_new_data[idx, :, :]

class InversePendulumProblemLearning():

    def __init__(self, problem):
        self.invpend = problem

        self.generate_data()

        return
    
    def generate_data(self):
        n_rollouts = 5

        x = torch.randn(n_rollouts, problem.T, 2, requires_grad=False)
        u = torch.randn(n_rollouts, problem.T-1, 1, requires_grad=False) + torch.sin(torch.linspace(0, np.pi, problem.T-1))[None, :, None]

        for i in range(n_rollouts):
            x[i, 0, 1] = 0.0    #zero initial angular velocity
            for t in range(problem.T-1):
                x[i, t+1] = torch.Tensor(problem.plant_dyn(x[i, t].numpy(), u[i, t].numpy(), t, None))

        self.dataset = InvertedPendulumDataset(x, u)

        return

    def build_nn_model(self):
        self.rnn_model = SimpleRNNModel(5)
        self.fnn_model = SimpleFNNModel(5)
        return
    
    def learn_nn_model(self, model='fnn'):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        
        n_epoches = 200

        if model == 'fnn':
            optim = torch.optim.Adam(self.fnn_model.parameters(), lr=1e-3, weight_decay=0.02)
        else:
            optim = torch.optim.Adam(self.rnn_model.parameters(), lr=1e-3, weight_decay=0.02)

        for e in range(n_epoches):
            avg_cost = []
            for batch_idx, (x, u, x_new) in enumerate(dataloader):
                #print(x.shape, u.shape, x_new.shape)
                optim.zero_grad()
                if model == 'fnn':
                    x_pred = self.fnn_model(x, u)
                else:
                    x_pred, hidden = self.rnn_model(x, u)
                #print(x_new.shape, x_pred.shape)
                cost = torch.mean(torch.norm(x_new-x_pred, dim=2))
                #print('Epoch {}/Batch {}: Training Error- {}'.format(e, batch_idx, cost.item()))
                avg_cost.append(cost.item())
                cost.backward()

                optim.step()
            if (e+1) % 10 == 0:
                print('Epoch {}: Training Error- {}'.format(e, np.mean(avg_cost)))
        return
    
    def solve_learned_ilqr_problem(self, x0):
        return
    
    def visualize_data(self):
        fig = plt.figure(figsize=(16, 8), dpi=80)
        ax_udata = fig.add_subplot(121)
        ax_xdata = fig.add_subplot(122)
        
        for i in range(len(self.dataset)):

            t = self.dataset[i][1].shape[0]

            ax_udata.plot(np.arange(t), self.dataset[i][1].numpy().flatten())
            ax_xdata.plot(np.arange(t), self.dataset[i][0][:, 0].numpy().flatten())

        plt.show()
        return

if __name__ == '__main__':
    problem = InversePendulumProblem()
    x0 = np.array([np.pi-1 , 0])
    
    # problem.build_ilqr_problem(grad_type=0) #try real gradients/hessians
    # problem.solve_ilqr_problem(x0)
    # problem.plot_ilqr_result()

    # problem.build_ilqr_problem(grad_type=1) #try autograd
    # problem.solve_ilqr_problem(x0)
    # problem.plot_ilqr_result()

    # problem.build_ilqr_problem(grad_type=2) #try finite difference
    # problem.solve_ilqr_problem(x0)
    # problem.plot_ilqr_result()

    learning = InversePendulumProblemLearning(problem)

    #learning.visualize_data()

    learning.build_nn_model()
    learning.learn_nn_model(model='rnn')