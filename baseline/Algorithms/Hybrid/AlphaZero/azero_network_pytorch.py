import functools
import operator
import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


class History:
    def __init__(self, history):
        self.history = history


class MakeDataset(Dataset):
    """
    """

    def __init__(self, states, policy, values, device):
        self.states = torch.from_numpy(states).float().to(device, non_blocking=True)
        self.policy = torch.from_numpy(policy).float().to(device, non_blocking=True)
        self.values = torch.from_numpy(values).float().to(device, non_blocking=True)
        self.len = states.shape[0]

    def __getitem__(self, index):
        return self.states[index], self.policy[index], self.values[index]

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


class AzeroBrain(nn.Module):
    def __init__(self, input_dim, output_dim, network_type="FC", lr=0.005, scope_name="", pv_loss_ratio=1,
                 use_gpu=False, num_layers=2, num_hidden=256):
        self.use_gpu = use_gpu
        cuda = torch.cuda.is_available() and use_gpu
        self.device = torch.device('cuda' if cuda else 'cpu')
        if not cuda:
            torch.set_num_threads(10)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.pv_loss_ratio = pv_loss_ratio
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        if scope_name == "":
            # np.random.seed()
            seed = np.random.randint(10000)
            scope_name = "worker_" + str(seed)
        self.scope_name = scope_name
        self.batch_norms = []
        self.conv_layers = []
        self.common_layers = []
        self.policy_layers = []
        self.value_layers = []
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=-1)
        self.network_type = network_type
        if network_type == "FC":
            self.compile_fully()
        if network_type == "FCPPO":
            self.compile_fully_PPO()
        if network_type == "QC":
            self.compile_fully_qc()
        if network_type == "QC2":
            self.compile_fully_qc2()
        if network_type == "CNN":
            assert len(input_dim) == 3, "Need 3 dimensional states to apply convolutions"
            self.compile_cnn_2()
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def compile(self, common_sizes, policy_sizes, value_sizes, in_size=None, return_policy=True):
        """

        Args:
            common_sizes:
            policy_sizes:
            value_sizes:
            in_size:

        Returns:

        """
        if in_size is None:
            in_size = self.input_dim[0]
        for i, next_size in enumerate(common_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("_fc{}".format(i), fc)  # self.scope_name +
            self.common_layers.append(fc)
        in_size_value = in_size
        for i, next_size in enumerate(policy_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("_fc_policy{}".format(i), fc)
            self.policy_layers.append(fc)
        in_size = in_size_value
        for i, next_size in enumerate(value_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("_fc_value{}".format(i), fc)
            self.value_layers.append(fc)
        if return_policy:
            out_1 = self.policy = nn.Linear(policy_sizes[-1], self.output_dim)
            self.__setattr__("_policy{}", out_1)
        else:
            out_1 = policy_sizes[-1]
        out_2 = self.value = nn.Linear(value_sizes[-1], 1)
        self.__setattr__("_value{}", out_2)
        return out_1, out_2

    def compile_fully_qc(self):
        """

        Returns:

        """
        sizes = [16]
        value_sizes = [4]
        policy_sizes = [8]
        self.activation = self.selu
        return self.compile(common_sizes=sizes, policy_sizes=policy_sizes, value_sizes=value_sizes)

    def compile_fully_qc2(self):
        """

        Returns:

        """
        sizes = [64, 32, 16]
        value_sizes = [8]
        policy_sizes = [16]
        self.activation = self.selu
        return self.compile(common_sizes=sizes, policy_sizes=policy_sizes, value_sizes=value_sizes)

    def compile_fully(self):
        """

        Returns:

        """
        sizes = [16]
        value_sizes = [16]
        policy_sizes = [16]
        # sizes = [self.num_hidden] * (self.num_layers - 1)
        # value_sizes = [self.num_hidden]
        # policy_sizes = [self.num_hidden]
        self.activation = self.relu
        return self.compile(common_sizes=sizes, policy_sizes=policy_sizes, value_sizes=value_sizes)

    def compile_fully_PPO(self):
        """

        Returns:

        """
        sizes = [128, 128]
        value_sizes = [4]
        policy_sizes = [10]
        self.activation = self.selu
        return self.compile(common_sizes=sizes, policy_sizes=policy_sizes, value_sizes=value_sizes)

    def compile_cnn_2(self):
        """

        Returns:

        """
        in_size = self.input_dim[0]
        strides = [1, 1, 2]  # , 1, 1, 2
        units = [64, 64, 64]
        self.activation = self.relu
        # strides = [1, 1, 2, 1, 1, 2]
        # units = [32, 64, 64, 64, 64, 64]
        for i in range(len(strides)):
            conv = nn.Conv2d(in_size, units[i], kernel_size=3, stride=strides[i])
            batch = nn.BatchNorm2d(units[i])
            self.batch_norms.append(batch)
            self.__setattr__("_batch_norm{}".format(i), batch)
            in_size = units[i]
            self.__setattr__("_conv{}".format(i), conv)
            self.conv_layers.append(conv)
        h = torch.rand(1, *self.input_dim)
        for i, conv in enumerate(self.conv_layers):
            h = self.relu(self.batch_norms[i](conv(h)))
        num_features_before_fcnn = functools.reduce(operator.mul,
                                                    list(h.shape))
        common_sizes = [128]
        policy_sizes = [128, 64]
        value_sizes = [128, 64]
        self.compile(common_sizes=common_sizes, policy_sizes=policy_sizes, value_sizes=value_sizes,
                     in_size=num_features_before_fcnn)

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
        y1 = np.asarray([s[1] for s in data])  # probability
        y2 = np.asarray([s[2] for s in data])  # value
        return self.fit(x, y1, y2, epochs, batch_size, stopping, verbose=verbose)

    def fit(self, states, policies, values, epochs, batch_size, stopping, verbose=0):
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
        policies = np.reshape(policies, (-1, self.output_dim))
        values = np.reshape(values, (-1, 1))
        dataset = MakeDataset(states, policies, values, self.device)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        history = {"loss": [],
                   "val_loss": [],
                   "probabilities_loss": [],
                   "value_loss": [],
                   "val_probabilities_loss": [],
                   "val_value_loss": []}
        for index_epoch in range(epochs):
            avg_loss = total_loss = 0
            avg_pol_loss = total_pol_loss = 0
            avg_value_loss = total_value_loss = 0
            count = 0
            step = 1
            for state, policy, value in data_loader:
                count += 1
                loss, policy_loss, value_loss = self._fit(state,
                                                          policy,
                                                          value)
                if np.isnan(loss):
                    print("What")
                total_loss += loss
                avg_loss = total_loss / (step + 1)

                total_pol_loss += policy_loss
                avg_pol_loss = total_pol_loss / (step + 1)

                total_value_loss += value_loss
                avg_value_loss = total_value_loss / (step + 1)
                step += 1
            avg_loss_val = avg_pol_loss_val = avg_value_loss_val = 0
            history['loss'].append(avg_loss)
            history['val_loss'].append(avg_loss_val)
            history['probabilities_loss'].append(avg_pol_loss)
            history['value_loss'].append(avg_value_loss)
            history['val_probabilities_loss'].append(avg_pol_loss_val)
            history['val_value_loss'].append(avg_value_loss_val)
            if verbose > 0:
                print("Epoch ", index_epoch, ": loss:", avg_loss, "; val_los:", avg_loss_val, "; pol_loss:",
                      avg_pol_loss, "; pol_val_loss:", avg_pol_loss_val, "; value_loss:", avg_value_loss,
                      "; val_value_loss:", avg_value_loss_val)
        return History(history)

    def fit_(self, states, policy, value, epochs, batch_size, stopping, verbose=0):
        """

        Args:
            states:
            policy:
            value:
            epochs:
            batch_size:
            stopping:
            verbose:

        Returns:

        """
        states = np.reshape(states, (-1,) + self.input_dim)
        policy = np.reshape(policy, (-1, self.output_dim))
        value = np.reshape(value, (-1, 1))

        # Reserve samples for validation.
        val_samples = int(0.2 * states.shape[0])

        s_val = states[-val_samples:]
        p_val = policy[-val_samples:]
        v_val = value[-val_samples:]
        s_train = states[:-val_samples]
        p_train = policy[:-val_samples]
        v_train = value[:-val_samples]

        history = {"loss": [],
                   "val_loss": [],
                   "probabilities_loss": [],
                   "value_loss": [],
                   "val_probabilities_loss": [],
                   "val_value_loss": []}
        for epoch in range(epochs):
            # print("\nStart of epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.
            avg_loss = total_loss = 0
            avg_pol_loss = total_pol_loss = 0
            avg_value_loss = total_value_loss = 0
            step = 0
            base = 0
            s_train, p_train, v_train = unison_shuffled(s_train, p_train, v_train)
            count = 0
            while base < s_train.shape[0] - batch_size:
                count += 1
                base = batch_size * step

                loss, policy_loss, value_loss = self._fit(s_train[base: base + batch_size],
                                                          p_train[base: base + batch_size],
                                                          v_train[base: base + batch_size],
                                                          transfer=True)
                if np.isnan(loss):
                    print("What")
                total_loss += loss
                avg_loss = total_loss / (step + 1)

                total_pol_loss += policy_loss
                avg_pol_loss = total_pol_loss / (step + 1)

                total_value_loss += value_loss
                avg_value_loss = total_value_loss / (step + 1)
                step += 1
                # print("Epoch:", epoch, "; step:", step + 1, "loss:", loss)
            avg_loss_val = total_loss = 0
            avg_pol_loss_val = total_pol_loss = 0
            avg_value_loss_val = total_value_loss = 0
            step = 0
            base = 0
            while base < s_val.shape[0] - batch_size:
                base = batch_size * step
                loss, policy_loss, value_loss = self._evaluate(s_val[base: base + batch_size],
                                                               p_val[base: base + batch_size],
                                                               v_val[base: base + batch_size],
                                                               transfer=True)
                total_loss += loss
                avg_loss_val = total_loss / (step + 1)

                total_pol_loss += policy_loss
                avg_pol_loss_val = total_pol_loss / (step + 1)

                total_value_loss += value_loss
                avg_value_loss_val = total_value_loss / (step + 1)
                step += 1
            history['loss'].append(avg_loss)
            history['val_loss'].append(avg_loss_val)
            history['probabilities_loss'].append(avg_pol_loss)
            history['value_loss'].append(avg_value_loss)
            history['val_probabilities_loss'].append(avg_pol_loss_val)
            history['val_value_loss'].append(avg_value_loss_val)
            if verbose > 0:
                print("Epoch ", epoch, ": loss:", avg_loss, "; val_los:", avg_loss_val, "; pol_loss:", avg_pol_loss,
                      "; pol_val_loss:", avg_pol_loss_val, "; value_loss:", avg_value_loss,
                      "; val_value_loss:", avg_value_loss_val)
        return History(history)

    def _fit(self, states, policy, value, update=True, transfer=False):
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
            policy = torch.from_numpy(policy).float().to(self.device)
            value = torch.from_numpy(value).float().to(self.device)
        self.optimizer.zero_grad()
        P, v = self(states)
        loss_policy = torch.mean(-policy * torch.log(P + 1e-07), dim=-1)
        value_loss = (value - v) ** 2
        loss = torch.mean(self.pv_loss_ratio * loss_policy + value_loss)
        if update:
            loss.backward()
            self.optimizer.step()
        return loss.item(), torch.mean(loss_policy).item(), torch.mean(value_loss).item()

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
        P, v = self.forward(x)
        return get_numpy(P), get_numpy(v)

    def predict_one(self, x):
        """

        Args:
            x:

        Returns:

        """
        P, v = self.predict(x[None])
        return P[0], v[0][0]

    def set_weights(self, weights):
        """

        Args:
            weights:

        Returns:

        """
        self.load_state_dict(weights)

    def get_weights(self, ):
        """

        Returns:

        """
        state_dict = self.state_dict()
        for param_tensor in state_dict:
            state_dict[param_tensor] = state_dict[param_tensor].to('cpu')
        return state_dict

    def save(self, path):
        """

        Args:
            path:

        Returns:

        """

        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), path + "/model.h5")

    def load(self, load_path):
        """

        Args:
            load_path:

        Returns:

        """
        self.load_state_dict(torch.load(load_path + "/model.h5", map_location=self.device))

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

    def forward(
            self,
            obs
    ):
        """

        Args:
            obs: Observation

        Returns:

        """
        h = obs.to(self.device)
        for i, conv in enumerate(self.conv_layers):
            h = self.relu(self.batch_norms[i](conv(h)))
        if len(self.conv_layers) > 0:
            h = h.view(h.size(0), -1)  ##  flatten

        for i, fc in enumerate(self.common_layers):
            h = self.activation(fc(h))
        h_v = h
        for i, fc in enumerate(self.policy_layers):
            h = self.activation(fc(h))
        h_p = h
        h = h_v
        for i, fc in enumerate(self.value_layers):
            h = self.activation(fc(h))
        h_v = h

        policy = self.softmax(self.policy(h_p))
        value = self.value(h_v)
        return policy, value

    def close(self):
        """parallel sampler calls brain.close()"""
        pass


if __name__ == '__main__':
    model = AzeroBrain(input_dim=(8,), output_dim=3)
    model2 = AzeroBrain(input_dim=(8,), output_dim=3)
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
