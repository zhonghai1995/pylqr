import numpy as np
import matplotlib.pyplot as plt

from pylqr_test import InversePendulumProblem

try:
    import torch
    from torch.utils.data import Dataset, DataLoader

except ImportError:
    print('Error in importing PyTorch package.')

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.normal_(m.bias.data)
    # if isinstance(m, torch.nn.GRU):
    #     for param in m.parameters():
    #         if len(param.shape) >= 2:
    #             torch.nn.init.orthogonal_(param.data)
    #         else:
    #             torch.nn.init.normal_(param.data)
            
    return

#so far, getting jacobian from pytorch needs to run backward gradient multiple times
#the following gist use batch input to alleviate this
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
#for now we dont need this to be differentiable as well, otherwise we need to retain_graph=True
def get_jacobian(func, x, n_outputs):
    x = x.squeeze()
    #note this will automatically expand dimension for x.shape = (dim,)
    x = x.repeat(n_outputs, 1)
    x.requires_grad_(True)
    y = func(x)
    y.backward(torch.eye(n_outputs), retain_graph=False)

    return x.grad.data

class SimpleRNNModel(torch.nn.Module):
    def __init__(self, n_rnncell):
        super(SimpleRNNModel, self).__init__()
        self.gru = torch.nn.GRU(n_rnncell, n_rnncell, batch_first=True)
        self.in_feat = torch.nn.Sequential(torch.nn.Linear(3, n_rnncell), torch.nn.ReLU())  #3: x-2, u-1
        self.out_feat = torch.nn.Linear(n_rnncell, 2)                                       #2: x-2
        self.n_rnncell = n_rnncell
        
        self.in_feat.apply(init_weights)
        init_weights(self.out_feat)
        init_weights(self.gru)

    def step(self, x, u, hidden):
        #x, u --> (batch, size)
        #hidden --> (batch, size)
        #keep this input and output layout for dynamics evaluation
        if hidden is not None:
            hidden = hidden[None, :, :]
        output, hidden = self.gru(self.in_feat(torch.cat((x, u), dim=1))[:, None, :], hidden)
        return self.out_feat(output)[:, 0, :], hidden[0, :, :]
    
    def forward(self, x_seq, u_seq, hidden=None):
        #batch x len x dim
        T = u_seq.size()[1]
        x_new_seq = []
        hidden_new_seq = []

        for i in range(T):
            x_new, hidden_new = self.step(x_seq[:, i, :], u_seq[:, i, :], hidden)       
            x_new_seq.append(x_new.clone())
            hidden_new_seq.append(hidden_new.clone())
            hidden = hidden_new
            #print(hidden[0].shape, hidden[1].shape)
        
        #combine seq and remember to swap the axes
        x_new_seq = torch.stack(x_new_seq, dim=0).transpose(0, 1)
        hidden_new_seq = torch.stack(hidden_new_seq, dim=0).transpose(0, 1)
        #print(x_new_seq.shape, hidden_new_seq.shape)
        return x_new_seq, hidden_new_seq

class SimpleFNNModel(torch.nn.Module):
    def __init__(self, n_latent):
        super(SimpleFNNModel, self).__init__()

        self.n_latent = n_latent

        self.in_feat = torch.nn.Sequential(
            torch.nn.Linear(3, n_latent),
            torch.nn.ReLU())
        self.out_feat = torch.nn.Linear(n_latent, 2)

        self.in_feat.apply(init_weights)
        init_weights(self.out_feat)
    
    def step(self, x, u):
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

    def __init__(self, problem, T=150):
        self.invpend = problem

        self.data_T = T
        self.generate_data()

        return
    
    def generate_data(self):
        n_rollouts = 10

        x = torch.randn(n_rollouts, self.data_T, 2, requires_grad=False)
        u = torch.randn(n_rollouts, self.data_T-1, 1, requires_grad=False) + torch.sin(torch.linspace(0, np.pi, self.data_T-1))[None, :, None]

        for i in range(n_rollouts):
            x[i, 0, 1] = 0.0    #zero initial angular velocity
            for t in range(self.data_T-1):
                x[i, t+1] = torch.Tensor(problem.plant_dyn(x[i, t].numpy(), u[i, t].numpy(), t, None))

        self.dataset = InvertedPendulumDataset(x, u)

        return

    def build_nn_model(self):
        self.rnn_model = SimpleRNNModel(2)
        self.fnn_model = SimpleFNNModel(5)
        return
    
    def learn_nn_model(self, model='fnn'):
        if model == 'fnn':
            n_epoches = 200
            dataloader = DataLoader(self.dataset, batch_size=5, shuffle=True)
            optim = torch.optim.Adam(self.fnn_model.parameters(), lr=1e-2, weight_decay=0.02)
        else:
            n_epoches = 1000
            dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)
            optim = torch.optim.Adam(self.rnn_model.parameters(), lr=5e-3, weight_decay=0.0)

        for e in range(n_epoches):
            avg_cost = []
            for _, (x, u, x_new) in enumerate(dataloader):
                #print(x.shape, u.shape, x_new.shape)
                optim.zero_grad()
                if model == 'fnn':
                    x_pred = self.fnn_model(x, u)
                else:
                    x_pred, _ = self.rnn_model(x, u)
                #print(x_new.shape, x_pred.shape)
                cost = torch.mean(torch.norm(x_new-x_pred, dim=2))
                #print('Epoch {}/Batch {}: Training Error- {}'.format(e, batch_idx, cost.item()))
                avg_cost.append(cost.item())
                cost.backward()

                optim.step()
            if (e+1) % 50 == 0:
                print('Epoch {}: Training Error- {}'.format(e, np.mean(avg_cost)))
        return
    
    def solve_learned_ilqr_problem(self, x0, u_init, n_itrs):
        self.invpend.solve_ilqr_problem(x0=x0, u_init=u_init, n_itrs=n_itrs, verbose=True)
        return self.invpend.res['u_array_opt'], self.invpend.res['x_array_opt']
    
    def solve_mpc_with_learned_model(self, model='fnn'):
        #prepare problem gradient and hessians
        self.invpend.build_ilqr_problem(grad_type=0)    #using closed form gradient and hessian of cost function
        #specify gradient for dynamic constraint
        if model == 'fnn':
            n_output_dim = 2
            def plant_dyn_dx(x, u, t, aux):
                #fnn, just input state...
                #remember to repeat variable that is not taking differentiation
                return get_jacobian(lambda x_for_jac: self.fnn_model.step(x_for_jac, torch.Tensor(u)[None, :].repeat(n_output_dim, 1)), torch.Tensor(x), n_outputs=2).numpy()
            def plant_dyn_du(x, u, t, aux):
                return get_jacobian(lambda u_for_jac: self.fnn_model.step(torch.Tensor(x)[None, :].repeat(n_output_dim, 1), u_for_jac), torch.Tensor(u), n_outputs=n_output_dim).numpy()
            self.real_plant_dyn = self.invpend.plant_dyn
            self.invpend.ilqr.plant_dyn = lambda x, u, t, aux: self.fnn_model.step(torch.Tensor(x)[None, :], torch.Tensor(u)[None, :])[0].data.numpy()
            self.invpend.ilqr.plant_dyn_dx = plant_dyn_dx
            self.invpend.ilqr.plant_dyn_du = plant_dyn_du
        else:
            #for rnn, output_dim include hidden state from the cell as well
            n_output_dim = 2 + self.rnn_model.n_rnncell
            get_cell_tensor = lambda x: x[:, 2:n_output_dim]

            def plant_dyn_dx(x, u, t, aux):
                def rnn_model_eval_x_for_jac(x_for_jac):
                    out, hidden = self.rnn_model.step(x_for_jac[:, :2],                                             #x 
                        torch.Tensor(u)[None, :].repeat(n_output_dim, 1),                                           #u
                        get_cell_tensor(x_for_jac))                                                                 #hidden
                    return torch.cat((out, hidden), dim=1)
                return get_jacobian(rnn_model_eval_x_for_jac, torch.Tensor(x), n_outputs=n_output_dim).numpy()

            def plant_dyn_du(x, u, t, aux):
                x_tensor = torch.Tensor(x).repeat(n_output_dim, 1)
                def rnn_model_eval_u_for_jac(u_for_jac):
                    out, hidden = self.rnn_model.step(x_tensor[:, :2],                                              #x 
                        u_for_jac,                                                                                  #u
                        get_cell_tensor(x_tensor))                                                                  #hidden
                    return torch.cat((out, hidden), dim=1)
                return get_jacobian(rnn_model_eval_u_for_jac, torch.Tensor(u), n_outputs=n_output_dim).numpy()

            self.real_plant_dyn = self.invpend.plant_dyn
            def rnn_plant_dyn(x, u, t, aux):
                #print(x.shape, u.shape)
                x_new, hidden_new = self.rnn_model.step(torch.Tensor(x[:2])[None, :], torch.Tensor(u)[None, :], 
                    torch.Tensor(x[2:n_output_dim])[None, :])
                return torch.cat((x_new, hidden_new), dim=1)[0].data.numpy()

            self.invpend.ilqr.plant_dyn = rnn_plant_dyn
            self.invpend.ilqr.plant_dyn_dx = plant_dyn_dx
            self.invpend.ilqr.plant_dyn_du = plant_dyn_du

            #remember cost gradient and hessian needs to be augmented with zero matrices corresponding to hidden state of rnn cell
            self.invpend.ilqr.cost_dx = lambda x, u, t, aux: np.concatenate([self.invpend.cost_dx(x,u,t,aux), np.zeros(self.rnn_model.n_rnncell)])
            self.invpend.ilqr.cost_dxx = lambda x, u, t, aux: np.block([    [self.invpend.cost_dxx(x,u,t,aux), np.zeros((2, self.rnn_model.n_rnncell))], 
                                                                            [np.zeros((self.rnn_model.n_rnncell, 2)), np.zeros((self.rnn_model.n_rnncell, self.rnn_model.n_rnncell))]])
            self.invpend.ilqr.cost_dux = lambda x, u, t, aux: np.concatenate([self.invpend.cost_dux(x,u,t,aux), np.zeros(self.rnn_model.n_rnncell)])


        if model == 'fnn':
            x0 = np.array([np.random.rand()-.5, 0.0])
        else:
            #for rnn, the state is concatenating x, h and c
            x0 = np.array([np.random.rand()-.5, 0.0]+[0]*self.rnn_model.n_rnncell)

        u_init = None
        state_traj = [x0]
        for i in range(self.data_T):
            #get mpc control
            u_array, x_array = self.solve_learned_ilqr_problem(state_traj[-1], u_init, 20)
            #apply the control to real dynamics
            if model == 'fnn':
                x_new = self.real_plant_dyn(state_traj[-1], u_array[0], None, None)
            else:
                x_new = self.real_plant_dyn(state_traj[-1][:2], u_array[0], None, None)
                #x_new = np.concatenate([x_new, np.zeros(self.rnn_model.n_rnncell)])       #for next state, zero hidden cell for a new rnn evaluation, the thing is real dynamics doesnt really depend on it here...
                x_new = np.concatenate([x_new, x_array[1][2:]])                          #or inherit predicted hidden?

            #record new state and previous control traj for a warm start...
            state_traj.append(x_new)
            u_init = u_array
            #print('Control step {}'.format(i))
        return np.array(state_traj)
    
    def visualize_data(self, model='fnn'):
        fig = plt.figure(figsize=(16, 8), dpi=80)
        ax_udata = fig.add_subplot(121)
        ax_xdata = fig.add_subplot(122)
        
        for i in range(len(self.dataset)):
            t = self.dataset[i][1].shape[0]

            ax_udata.plot(np.arange(t), self.dataset[i][1].numpy().flatten())
            ax_xdata.plot(np.arange(t), self.dataset[i][0][:, 0].numpy().flatten())

            #note this is not recursive prediction, just see how well the data fits
            if model == 'fnn':
                x_pred_fnn = self.fnn_model(self.dataset[i][0][None, :, :], self.dataset[i][1][None, :, :])
                ax_xdata.plot(np.arange(t), x_pred_fnn[0, :, 0].detach().numpy().flatten(), '.-')
            else:
                x_pred_rnn, _ = self.rnn_model(self.dataset[i][0][None, :, :], self.dataset[i][1][None, :, :])
                ax_xdata.plot(np.arange(t), x_pred_rnn[0, :, 0].detach().numpy().flatten(), '.-')
        plt.show()
        return

if __name__ == '__main__':
    #T for MPC time horizon
    problem = InversePendulumProblem(T=5)
    x0 = np.array([np.pi-1 , 0])
    model = 'fnn'

    #T for data sequence length
    learning = InversePendulumProblemLearning(problem, T=150)

    learning.build_nn_model()
    learning.learn_nn_model(model=model)

    learning.visualize_data(model=model)

    print('Apply MPC with learned model...')

    ctrl_traj = learning.solve_mpc_with_learned_model(model=model)
    #draw the phase plot
    fig = plt.figure(figsize=(16, 8), dpi=80)
    ax_phase = fig.add_subplot(111)
    theta = ctrl_traj[:, 0]
    theta_dot = ctrl_traj[:, 1]
    ax_phase.plot(theta, theta_dot, 'k', linewidth=3.5)
    ax_phase.set_xlabel('theta (rad)', fontsize=20)
    ax_phase.set_ylabel('theta_dot (rad/s)', fontsize=20)
    ax_phase.set_title('Phase Plot', fontsize=20)

    ax_phase.plot([theta[-1]], [theta_dot[-1]], 'b*', markersize=16)

    plt.show()