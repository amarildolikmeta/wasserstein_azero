import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from .networks import FlattenMlp

def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


class History:
    def __init__(self, history):
        self.history = history


class MakeDataset(Dataset):
    """
    """

    def __init__(self, states, actions, values, stds, device):
        self.states = torch.from_numpy(states).float().to(device, non_blocking=True)
        self.actions = torch.from_numpy(actions).float().to(device, non_blocking=True)
        self.values = torch.from_numpy(values).float().to(device, non_blocking=True)
        self.stds = torch.from_numpy(stds).float().to(device, non_blocking=True)
        self.len = states.shape[0]

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.values[index], self.stds[index]

    def __len__(self):
        return self.len


def unison_shuffled(a, b, c):
    """
    Args:
        a:
        b:
        c:
    Returns:

    """
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


class AzeroWassersteinBrain(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, num_hidden=256, lr=0.005, scope_name="", use_gpu=False,
                 init_mean=0., init_std=1.,  prv_std_qty=0., prv_std_weight=1.):
        self.use_gpu = use_gpu
        cuda = torch.cuda.is_available() and use_gpu
        self.device = torch.device('cuda' if cuda else 'cpu')
        if not cuda:
            torch.set_num_threads(10)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.prv_std_qty = prv_std_qty
        self.prv_std_weight = prv_std_weight
        if scope_name == "":
            # np.random.seed()
            seed = np.random.randint(10000)
            scope_name = "worker_" + str(seed)
        self.scope_name = scope_name
        self.batch_norms = []
        self.conv_layers = []
        self.common_layers = []
        self.std_layers = []
        self.value_layers = []
        self.init_std = init_std
        self.init_log_std = np.log(init_std)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=-1)
        # self.network_type = network_type
        # if network_type == "FC":
        #     self.compile_fully()
        # if network_type == "FCPPO":
        #     self.compile_fully_PPO()
        # if network_type == "QC":
        #     self.compile_fully_qc()
        # if network_type == "QC2":
        #     self.compile_fully_qc2()
        # if network_type == "CNN":
        #     assert len(input_dim) == 3, "Need 3 dimensional states to apply convolutions"
        #     self.compile_cnn_2()
        hidden_sizes = [num_hidden] * num_layers
        self.q = FlattenMlp(input_size=input_dim[0],
                            output_size=output_dim,
                            hidden_sizes=hidden_sizes,
                            bias=init_mean,
                            positive=False,
                            train_bias=True)
        self.std = FlattenMlp(input_size=input_dim[0],
                            output_size=output_dim,
                            hidden_sizes=hidden_sizes,
                            bias=init_std,
                            positive=True,
                            train_bias=True)

        self.old_std = FlattenMlp(input_size=input_dim[0],
                            output_size=output_dim,
                            hidden_sizes=hidden_sizes,
                            bias=init_std,
                            positive=True,
                            train_bias=True)

        self.q.to(self.device)
        self.std.to(self.device)
        self.old_std.to(self.device)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.std_optimizer = optim.Adam(self.std.parameters(), lr=lr)
        self.mse = nn.MSELoss()


    def train(self, data, epochs, batch_size, stopping=False, verbose=0):
        """

        Args:
            data:
            epochs:
            batch_size:
            stopping:
            verbose:

        Returns:

        """
        x = np.asarray([s[0] for s in data])  # state
        a = np.asarray([s[1] for s in data])  # action
        q = np.asarray([s[2] for s in data])  # value
        std = np.asarray([s[3] for s in data])  # std
        return self.fit(x, a, q, std, epochs, batch_size, stopping, verbose=verbose)

    def fit(self, states, actions, values, stds, epochs, batch_size, stopping, verbose=0):
        """

        Args:
            states:
            policies:
            values:
            epochs:
            batch_size:
            stopping:
            verbose:

        Returns:

        """
        states = np.reshape(states, (-1,) + self.input_dim)
        actions = np.reshape(actions, (-1, 1))
        values = np.reshape(values, (-1, 1))
        stds = np.reshape(stds, (-1, 1))
        dataset = MakeDataset(states, actions, values, stds, self.device)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        history = {"loss": [],
                   "val_loss": [],
                   "q_loss": [],
                   "std_loss": [],
                   "old_std_loss": [],
                   "val_q_loss": [],
                   "val_std_loss": [],
                   "mean_q": [],
                   "mean_std": []}
        if self.prv_std_qty > 0:
            self.old_std.load_state_dict(self.std.state_dict())

        for index_epoch in range(epochs):
            avg_loss = total_loss = 0
            avg_q = total_q = 0
            avg_std = total_std = 0
            avg_std_loss = total_std_loss = 0
            avg_old_std_loss = total_old_std_loss = 0
            avg_value_loss = total_value_loss = 0
            count = 0
            step = 1
            for state, action, value, std in data_loader:
                count += 1
                loss, q_loss, std_loss, mean_q, mean_std, old_std_loss = self._fit(state, action, value, std)
                if np.isnan(loss):
                    print("What")
                total_loss += loss
                avg_loss = total_loss / (step + 1)

                total_std_loss += std_loss
                avg_std_loss = total_std_loss / (step + 1)

                total_old_std_loss += old_std_loss
                avg_old_std_loss = total_old_std_loss / (step + 1)

                total_value_loss += q_loss
                avg_value_loss = total_value_loss / (step + 1)

                total_q += mean_q
                avg_q = total_q / (step + 1)

                total_std += mean_std
                avg_std = total_std / (step + 1)

                step += 1
            avg_loss_val = avg_pol_loss_val = avg_value_loss_val = 0
            history['loss'].append(avg_loss)
            history['val_loss'].append(avg_loss_val)
            history['std_loss'].append(avg_std_loss)
            history['q_loss'].append(avg_value_loss)
            history['old_std_loss'].append(avg_old_std_loss)
            history['mean_q'].append(avg_q)
            history['mean_std'].append(avg_std)
            if verbose > 0:
                print("Epoch ", index_epoch, ": loss:", avg_loss, "; val_loss:", avg_loss_val, "; q_loss:",

                      avg_value_loss, "; q_val_loss:", avg_value_loss_val, "; std_loss:", avg_std_loss,
                      "; val_std_loss:", avg_value_loss_val)
        return History(history)

    def _fit(self, states, action, value, stds, update=True, transfer=False):
        """

        Args:
            states:
            policy:
            value:
            update:
            transfer:

        Returns:

        """
        if transfer:
            states = torch.from_numpy(states).float().to(self.device)
            action = torch.from_numpy(action).float().to(self.device)
            value = torch.from_numpy(value).float().to(self.device)
            stds = torch.from_numpy(stds).float().to(self.device)
        self.q_optimizer.zero_grad()
        self.std_optimizer.zero_grad()
        q = self.q(states).gather(1, action.long())
        std = self.std(states).gather(1, action.long())
        value_loss = (value - q) ** 2
        std_loss = (std - stds) ** 2
        loss = torch.mean(value_loss + std_loss)
        value_loss = value_loss.mean()
        std_loss = std_loss.mean()
        if self.prv_std_qty > 0:
            qty = int(np.round(states.shape[0] * self.prv_std_qty))
            # qty = qty if qty > 0 else 1 # never irrelevant
            f_obs = torch.FloatTensor(qty, states.shape[1]).uniform_(-1, 1).to(self.device)
            f_actions = np.random.choice(self.output_dim, size=(qty, 1))
            f_actions = torch.from_numpy(f_actions).int().to(self.device)
            f_std1_preds = self.std(f_obs).gather(1, f_actions.long())
            f_std1_target = self.old_std(f_obs).gather(1, f_actions.long())
            f_std1_target = torch.clamp(f_std1_target, 0, self.init_std)
            old_std_loss = ((f_std1_preds - f_std1_target.detach()) ** 2).mean()
            std_loss += self.prv_std_weight * old_std_loss
        if update:
            value_loss.backward()
            self.q_optimizer.step()
            std_loss.backward()
            self.std_optimizer.step()

        return loss.item(), value_loss.item(), std_loss.item(), torch.mean(q).item(), torch.mean(std).item(),\
               old_std_loss.item()

    def _evaluate(self, states, policy, value, transfer=True):
        """

        Args:
            states:
            policy:
            value:
            transfer:

        Returns:

        """
        return self._fit(states, policy, value, update=False, transfer=transfer)

    def predict(self, x):
        """

        Args:
            x:

        Returns:

        """
        x = torch.from_numpy(x).float().to(self.device)
        q = self.q.forward(x)
        std = self.std.forward(x)
        return get_numpy(q), get_numpy(std)

    def predict_one(self, x):
        """

        Args:
            x:

        Returns:

        """
        q, std = self.predict(x[None])
        return q[0], std[0]

    def set_weights(self, weights):
        """
        Args:
            weights:

        Returns:
        """
        q_state_dict, std_state_dict, old_std_state_dict = weights[:]
        self.q.load_state_dict(q_state_dict)
        self.std.load_state_dict(std_state_dict)
        self.old_std.load_state_dict(old_std_state_dict)

    def get_weights(self, ):
        """
        Returns:
        """
        q_state_dict = self.q.state_dict()
        std_state_dict = self.std.state_dict()
        old_std_state_dict = self.old_std.state_dict()
        for param_tensor in q_state_dict:
            q_state_dict[param_tensor] = q_state_dict[param_tensor].to('cpu')
        for param_tensor in std_state_dict:
            std_state_dict[param_tensor] = std_state_dict[param_tensor].to('cpu')
        for param_tensor in old_std_state_dict:
            old_std_state_dict[param_tensor] = old_std_state_dict[param_tensor].to('cpu')
        weights = [q_state_dict, std_state_dict, old_std_state_dict]
        return weights

    def save(self, path):
        """

        Args:
            path:

        Returns:

        """
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.q.state_dict(), path + "/q.h5")
        torch.save(self.std.state_dict(), path + "/std.h5")
        torch.save(self.old_std.state_dict(), path + "/old_std.h5")

    def load(self, load_path):
        """

        Args:
            load_path:

        Returns:

        """
        q_load_path = load_path + "/q.h5"
        std_load_path = load_path + "/std.h5"
        old_std_load_path = load_path + "/old_std.h5"
        self.q.load_state_dict(torch.load(q_load_path, map_location=self.device))
        self.std.load_state_dict(torch.load(std_load_path, map_location=self.device))
        self.old_std.load_state_dict(torch.load(old_std_load_path, map_location=self.device))

    def load_weights(self, load_path):
        """

        Args:
            load_path:

        Returns:

        """
        self.load(load_path)

    def close(self):
        """

        Returns:

        """
        pass

    def close(self):
        """parallel sampler calls brain.close()"""
        pass


if __name__ == '__main__':
    model = AzeroWassersteinBrain(input_dim=(8,), output_dim=3)
    model2 = AzeroWassersteinBrain(input_dim=(8,), output_dim=3)
    weights = model.get_weights()
    model2.set_weights(weights)
    for i in range(100):
        state = np.random.random(8)
        P, v = model.predict_one(state)
        print("P:", P)
        print("V:", v)
        P, v = model2.predict_one(state)
        print("P2:", P)
        print("V2:", v)
