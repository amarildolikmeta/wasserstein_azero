import numpy as np
import scipy.stats as sts
import logging
import torch
from torch import nn as nn
from os import stat
from torch.distributions import Distribution, Normal
from torch.utils.data import Dataset, DataLoader

try:
    from .azero_network_pytorch import AzeroBrain, get_numpy, History
except:
    from azero_network_pytorch import AzeroBrain, get_numpy, History

# Use default (root) logger
logger = logging.getLogger()
train_logger = logging.getLogger("train")


class MakeDataset(Dataset):
    def __init__(self, states, policy, actions, values, device):
        if np.isnan(np.sum(states)) or np.isnan(np.sum(policy)) or np.isnan(np.sum(actions)) or np.isnan(
                np.sum(values)):
            logging.error("What")
            logging.error("np.sum(states): {}".format(np.sum(states)))
            logging.error("np.sum(policy): {}".format(np.sum(policy)))
            logging.error("np.sum(actions): {}".format(np.sum(actions)))
            logging.error("np.sum(values): {}".format(np.sum(values)))
        self.states = torch.from_numpy(states).float().to(device, non_blocking=True)
        self.policy = torch.from_numpy(policy).float().to(device, non_blocking=True)
        self.actions = torch.from_numpy(actions).float().to(device, non_blocking=True)
        self.values = torch.from_numpy(values).float().to(device, non_blocking=True)
        self.len = states.shape[0]

    def __getitem__(self, index):
        return self.states[index], self.policy[index], self.actions[index], self.values[index]

    def __len__(self):
        return self.len


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6, device=None):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        self.device = device
        self.epsilon = 0.999

    def sample_n(self, n, return_pre_tanh_value=False):
        """

        Args:
            n:
            return_pre_tanh_value:

        Returns:

        """
        z = self.normal.sample_n(n)
        x = torch.tanh(z)
        x = torch.clamp(x, -self.epsilon, self.epsilon)
        if return_pre_tanh_value:
            return x, z
        else:
            return x

    def log_prob(self, value, pre_tanh_value=None):
        """

        Args:
            value: some value, x
            pre_tanh_value: arctanh(x)

        Returns:

        """
        value = torch.clamp(value, -self.epsilon, self.epsilon)
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        Args:
            return_pretanh_value:

        Returns:

        """
        z = self.normal.sample().detach()
        x = torch.tanh(z)
        x = torch.clamp(x, -self.epsilon, self.epsilon)
        if return_pretanh_value:
            return x, z
        else:
            return x

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        Args:
            return_pretanh_value:

        Returns:

        """
        z = (
                self.normal_mean +
                self.normal_std *
                Normal(
                    torch.zeros(self.normal_mean.size(), device=self.device),
                    torch.ones(self.normal_std.size(), device=self.device)
                ).sample()
        )
        z.requires_grad_()
        x = torch.tanh(z)
        x = torch.clamp(x, -self.epsilon, self.epsilon)
        if return_pretanh_value:
            return x, z
        else:
            return x

    def entropy(self):
        k = self.normal_mean.size(-1)
        entropy = 0.5 * k * (1 + 2 * np.log(np.pi)) * torch.prod(self.normal_std, dim=-1)
        return entropy


class AzeroMixedBrain(AzeroBrain):
    def __init__(self, input_dim, output_dim, network_type="FC", lr=0.005, scope_name="", pv_loss_ratio=1,
                 use_gpu=False, std=None, entropy_coef=0.):
        self.std = std
        self.entropy_coef = entropy_coef
        self.n_estimators = output_dim[-1]
        super().__init__(input_dim, output_dim, network_type, lr, scope_name, pv_loss_ratio, use_gpu)

    def compile(self, common_sizes, policy_sizes, value_sizes, in_size=None, return_policy=True):
        """

        Args:
            common_sizes:
            policy_sizes:
            value_sizes:
            in_size:

        Returns:

        """
        policy_features, out_2 = super().compile(common_sizes, policy_sizes, value_sizes, in_size, return_policy=False)
        self.policies = []
        for i in range(self.n_estimators):
            parameters = nn.Linear(policy_features, self.output_dim[0])
            self.__setattr__("_policy_cont_{}".format(i), parameters)
            gate = nn.Linear(policy_features, self.output_dim[1])
            self.__setattr__("_policy_gate_{}".format(i), gate)
            target = nn.Linear(policy_features, self.output_dim[2])
            self.__setattr__("_policy_target_{}".format(i), target)
            if self.std is None:
                last_fc_log_std = nn.Linear(policy_features, self.output_dim[0])
                self.__setattr__("_log_std_{}".format(i), last_fc_log_std)
                policy = (parameters, gate, target, last_fc_log_std)
            else:
                if i == 0:
                    self.log_std = np.log(self.std)
                policy = (parameters, gate, target)
            self.policies.append(policy)
        return self.policies, out_2

    def predict(self, x):
        """

        Args:
            x:

        Returns:

        """
        x = torch.from_numpy(x).float().to(self.device)
        a, v, _, _, _, _, _ = self.forward(x)
        return get_numpy(a), get_numpy(v)

    def predict_one(self, x):
        """

        Args:
            x:

        Returns:

        """
        a, v = self.predict(x[None])
        return a[0], v[0][0]

    def predict_value(self, x):
        if len(x.shape) == 1:
            x = x[np.newaxis, ...]
        x = torch.from_numpy(x).float().to(self.device)
        _, v, _, _, _, _, _ = self.forward(x)
        return get_numpy(v)[0]

    def sample(self, x):
        if len(x.shape) == 1:
            x = x[np.newaxis, ...]
        x = torch.from_numpy(x).float().to(self.device)
        a, _, log_prob, _, _, _, _ = self.forward(x, deterministic=False)
        return get_numpy(a), get_numpy(log_prob)[0][0]

    def log_prob(self, x, a):
        if len(x.shape) == 1:
            x = x[np.newaxis, ...]
        x = torch.from_numpy(x).float().to(self.device)
        _, _, log_prob, means, log_stds, gate_log_probs, target_log_probs = self.forward(x)
        log_prob = 0
        means = get_numpy(means)[0]
        log_stds = get_numpy(log_stds)[0]
        gate_log_probs = get_numpy(gate_log_probs)[0]
        target_log_probs = get_numpy(target_log_probs)[0]
        a = a[0]
        for q in range(a.shape[0]):  # for each q_bit
            action = a[q][:-2]  # all expect gate and target
            gate = int(a[q][-2])
            target = int(a[q][-1])
            mean = means[q]
            log_std = log_stds[q]
            lp = np.log(sts.multivariate_normal(mean=mean, cov=np.diag(np.exp(log_std))).pdf(action) + 1e-10)
            gate_lp = gate_log_probs[q][gate]
            target_lp = target_log_probs[q][target]
            log_prob += lp + gate_lp + target_lp
        return log_prob

    def _fit(self, states, policy, action, value, update=True, transfer=False):
        if transfer:
            if np.isnan(np.sum(states)) or np.isnan(np.sum(policy)) or np.isnan(np.sum(action)) or np.isnan(
                    np.sum(value)):
                logging.error("What")
                logging.error("np.sum(states): {}".format(np.sum(states)))
                logging.error("np.sum(policy): {}".format(np.sum(policy)))
                logging.error("np.sum(action): {}".format(np.sum(action)))
                logging.error("np.sum(value): {}".format(np.sum(value)))
            states = torch.from_numpy(states).float().to(self.device)
            policy = torch.from_numpy(policy).float().to(self.device)
            action = torch.from_numpy(action).float().to(self.device)
            value = torch.from_numpy(value).float().to(self.device)
        self.optimizer.zero_grad()
        _, v, _, means, log_stds, gate_log_probs, target_log_probs = self(states)
        log_prob = 0
        entropy = 0

        # Debug only
        log_target = 0
        log_gate = 0
        log_lp = 0

        for q in range(action.shape[1]):  # for each q_bit
            log_std = log_stds[:, q, :]
            mean = means[:, q, :]
            std = torch.exp(log_std)
            a = action[:, q, :]
            gates = a[:, -2].to(dtype=torch.long, device=self.device)
            targets = a[:, -1].to(dtype=torch.long, device=self.device)
            a = a[:, :-2]
            tanh_normal = TanhNormal(mean, std, device=self.device)
            # action = torch.clamp(action, -0.9999999, 0.9999999)
            lp = tanh_normal.log_prob(a)
            lp = lp.sum(dim=1, keepdim=True)
            # log_prob = torch.clamp(log_prob, min=-1e-3, max=1e3) # numerical
            h = tanh_normal.entropy()
            gate_log_probs_ = gate_log_probs[:, q, :]
            gate_lp = gate_log_probs_[torch.arange(gates.size(0)), gates[:]]
            target_log_probs_ = target_log_probs[:, q, :]
            target_lp = target_log_probs_[torch.arange(gates.size(0)), targets[:]]
            log_prob += lp + gate_lp + target_lp # Qui c'è quello che devi separare per il plot
            entropy += h
            # Debug only
            log_target += target_lp
            log_lp += lp
            log_gate += gate_lp

        # Debug only
        loss_policy_target = torch.mean((log_target - torch.log(policy + 1e-10)) * log_target).item()
        loss_policy_gate = torch.mean((log_gate - torch.log(policy + 1e-10)) * log_gate).item()
        loss_policy_lp = torch.mean((log_lp - torch.log(policy + 1e-10)) * log_lp).item()

        loss_policy = (log_prob - torch.log(policy + 1e-10)) * log_prob
        entropy_loss = self.entropy_coef * entropy
        # loss_policy
        value_loss = (value - v) ** 2
        loss = torch.mean(self.pv_loss_ratio * (loss_policy - entropy_loss) + value_loss)
        if torch.isnan(loss):
            logger.error("Nan loss")
        if update:
            loss.backward()
            self.optimizer.step()
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if torch.isnan(torch.sum(param)):
                        logger.error("What: torch.sum(param) is Nan")

        policy_losses = (torch.mean(loss_policy).item(), loss_policy_target, loss_policy_gate, loss_policy_lp)

        return loss.item(), policy_losses, torch.mean(value_loss).item(), torch.mean(entropy_loss).item(),\
               torch.mean(entropy).item()

    def train(self, data, epochs, batch_size, stopping=False, verbose=0):
        x = np.asarray([s[0] for s in data])  # state
        y1 = np.asarray([s[1] for s in data])  # probability
        y2 = np.asarray([s[2] for s in data])  # action
        y3 = np.asarray([s[3] for s in data])  # value
        return self.fit(x, y1, y2, y3, epochs, batch_size, stopping, verbose=verbose)

    def validate(self, data, verbose=0, deterministic=False):
        x = np.asarray([s[0] for s in data])  # state
        y1 = np.asarray([s[1] for s in data])  # probability
        y2 = np.asarray([s[2] for s in data])  # action
        y3 = np.asarray([s[3] for s in data])  # value
        return self._validate(x, y1, y2, y3, verbose=verbose, deterministic=deterministic)

    def _validate(self, states, policies, actions, values, verbose=0, deterministic=False):
        from sklearn.metrics import r2_score
        states = np.reshape(states, (-1,) + self.input_dim)
        actions = np.reshape(actions, (-1, self.output_dim[-1], self.output_dim[0] + 2))
        policies = np.reshape(policies, (-1, 1))
        values = np.reshape(values, (-1, 1))
        states_ = torch.from_numpy(states).float().to(self.device, non_blocking=True)
        # policies = torch.from_numpy(policies).float().to(self.device, non_blocking=True)
        # actions = torch.from_numpy(actions).float().to(self.device, non_blocking=True)
        # values = torch.from_numpy(values).float().to(self.device, non_blocking=True)
        action, value, log_prob, means, log_stds, gate_log_probs, target_log_probs = self.forward(states_,
                                                                                                  deterministic=deterministic)
        rs = []
        acc_gates = []
        acc_targets = []
        action = action.detach().numpy()
        value = value.detach().numpy()
        for i in range(action.shape[1]):  # every q_bit
            a = action[:, i, :-2]
            a_target = actions[:, i, :-2]
            gate = action[:, i, -2]
            target = action[:, i, -1]
            gate_target = actions[:, i, -2]
            target_target = actions[:, i, -1]
            coefficient_of_dermination = r2_score(a, a_target)
            acc_gate = (gate == gate_target).sum() / states.shape[0]
            acc_target = (target == target_target).sum() / states.shape[0]
            rs.append(coefficient_of_dermination)
            acc_gates.append(acc_gate)
            acc_targets.append(acc_target)
        r2_value = r2_score(value, values)
        return rs, acc_gates, acc_targets, r2_value

    def fit(self, states, policies, actions, values, epochs, batch_size, stopping, verbose=0):
        states = np.reshape(states, (-1,) + self.input_dim)
        actions = np.reshape(actions, (-1, self.output_dim[-1], self.output_dim[0] + 2))
        policies = np.reshape(policies, (-1, 1))
        values = np.reshape(values, (-1, 1))
        dataset = MakeDataset(states, policies, actions, values, self.device)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        history = {"loss": [],
                   "val_loss": [],
                   "probabilities_loss": [],
                   "value_loss": [],
                   "val_probabilities_loss": [],
                   "val_value_loss": [],
                   "policy_target_loss": [],
                   "policy_gate_loss": [],
                   "policy_lp_loss": [],
                   "entropy_loss": [],
                   "entropy": [],
                   }
        for index_epoch in range(epochs):
            avg_loss = total_loss = 0
            avg_pol_loss = total_pol_loss = 0
            avg_value_loss = total_value_loss = 0
            # DEBUG only
            avg_policy_target_loss = total_policy_target_loss = 0
            avg_policy_gate_loss = total_policy_gate_loss = 0
            avg_policy_lp_loss = total_policy_lp_loss = 0
            avg_entropy_loss = total_entropy_loss = 0
            avg_entropy = total_entropy = 0
            count = 0
            step = 1
            for state, policy, action, value in data_loader:
                count += 1
                loss, policy_losses, value_loss, entropy_loss, entropy = self._fit(state,
                                                          policy,
                                                          action,
                                                          value)
                policy_loss, loss_policy_target, loss_policy_gate, loss_policy_lp = policy_losses

                if np.isnan(loss):
                    logger.error("Loss is Nan")
                total_loss += loss
                avg_loss = total_loss / (step + 1)

                total_pol_loss += policy_loss
                avg_pol_loss = total_pol_loss / (step + 1)

                total_value_loss += value_loss
                avg_value_loss = total_value_loss / (step + 1)

                #DEBUG only
                total_policy_target_loss += loss_policy_target
                avg_policy_target_loss = total_policy_target_loss / (step + 1)
                total_policy_gate_loss += loss_policy_gate
                avg_policy_gate_loss = total_policy_gate_loss / (step + 1)
                total_policy_lp_loss += loss_policy_lp
                avg_policy_lp_loss = total_policy_lp_loss / (step + 1)

                total_entropy_loss += entropy_loss
                avg_entropy_loss = total_entropy_loss / (step + 1)

                total_entropy += entropy
                avg_entropy = total_entropy / (step + 1)

                step += 1
            avg_loss_val = avg_pol_loss_val = avg_value_loss_val = 0
            history['loss'].append(avg_loss)
            history['val_loss'].append(avg_loss_val)
            history['probabilities_loss'].append(avg_pol_loss)
            history['value_loss'].append(avg_value_loss)
            history['val_probabilities_loss'].append(avg_pol_loss_val)
            history['val_value_loss'].append(avg_value_loss_val)
            history['policy_target_loss'].append(avg_policy_target_loss)
            history['policy_gate_loss'].append(avg_policy_gate_loss)
            history['policy_lp_loss'].append(avg_policy_lp_loss)
            history['entropy_loss'].append(avg_entropy_loss)
            history['entropy'].append(avg_entropy)

            logger.debug("Epoch ", index_epoch, ": loss:", avg_loss, "; val_los:", avg_loss_val, "; pol_loss:",
                  avg_pol_loss, "; pol_val_loss:", avg_pol_loss_val, "; value_loss:", avg_value_loss,
                  "; val_value_loss:", avg_value_loss_val)

            # Debug only
            path_log_file = train_logger.handlers[0].baseFilename
            if stat(path_log_file).st_size == 0:
                train_logger.debug(
                    "loss, val_probabilities_loss, value_loss, policy_target_loss, policy_gate_loss, policy_lp_loss, entropy_loss, entropy")
            train_logger.debug("{}, {}, {}, {}, {}, {}, {}, {}".format(avg_loss, avg_pol_loss,
                                                               avg_value_loss, avg_policy_target_loss,
                                                               avg_policy_gate_loss, avg_policy_lp_loss,
                                                               avg_entropy_loss, avg_entropy))
        return History(history)

    def forward(
            self,
            obs,
            deterministic=False,
            reparameterize=True
    ):
        """

        Args:
            obs: Observation
            deterministic:
            reparameterize:

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
        log_probs = []
        action = []
        means = []
        log_stds = []
        gate_log_probs = []
        target_log_probs = []
        for i in range(self.n_estimators):
            policy = self.policies[i]
            mean = policy[0](h_p)
            gate = self.softmax(policy[1](h_p))
            target = self.softmax(policy[2](h_p))

            if self.std is None:
                log_std = policy[3](h_p)
                std = torch.exp(log_std)
            else:
                std = self.std
                log_std = torch.log(std)
            log_prob = None
            if deterministic:
                parameter = torch.tanh(mean)
                chosen_gate = torch.argmax(gate, dim=-1, keepdim=True)
                chosen_target = torch.argmax(target, dim=-1, keepdim=True)
            else:
                tanh_normal = TanhNormal(mean, std, device=self.device)
                if reparameterize is True:
                    parameter, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    parameter, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    parameter,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
                idx = gate.multinomial(num_samples=1, replacement=True)
                chosen_gate = idx
                log_prob_gate = torch.log(gate[torch.arange(idx.size(0)), idx[:, 0]])
                log_prob_gate = torch.unsqueeze(log_prob_gate, 1)  # torch.unsqueeze(torch.log(gate[idx[:,0]]), 1)
                idx = target.multinomial(num_samples=1, replacement=True)
                chosen_target = idx
                log_prob_target = torch.log(
                    target[torch.arange(idx.size(0)), idx[:, 0]])  # torch.unsqueeze(torch.log(target[idx]), 1)
                log_prob_target = torch.unsqueeze(log_prob_target, 1)
                log_prob = log_prob + log_prob_target + log_prob_gate
            means.append(mean)
            log_stds.append(log_std)
            gate_log_probs.append(torch.log(gate))
            target_log_probs.append(torch.log(target))
            log_probs.append(log_prob)
            try:
                action.append(torch.cat([parameter, chosen_gate.float(), chosen_target.float()], dim=-1))
            except:
                logger.error("What")
        value = self.value(h_v)  # Batch_size, 1
        if not deterministic:
            log_probs = torch.stack(log_probs, dim=0)
            log_prob = torch.sum(log_probs, dim=1, keepdim=True)  # Batch_size, 1
        action = torch.stack(action, dim=0).permute(1, 0, 2)  # Batch_size, n_qubits, n_outputs (m + 2)
        means = torch.stack(means, dim=0).permute(1, 0, 2)  # Batch_size, n_qubits, n_params (m)
        log_stds = torch.stack(log_stds, dim=0).permute(1, 0, 2)  # Batch_size, n_qubits, n_params (m)
        gate_log_probs = torch.stack(gate_log_probs, dim=0).permute(1, 0, 2)  # Batch_size, n_qubits, n_gates (k)
        target_log_probs = torch.stack(target_log_probs, dim=0).permute(1, 0, 2)  # Batch_size, n_qubits, n_qubits (q)
        return (action, value, log_prob, means, log_stds, gate_log_probs, target_log_probs)

    def close(self):
        pass


if __name__ == '__main__':
    model = AzeroMixedBrain(input_dim=(8,), output_dim=3)
    model2 = AzeroMixedBrain(input_dim=(8,), output_dim=3)
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
