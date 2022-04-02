from copy import deepcopy

import numpy as np
import torch.nn
from matplotlib import cm

from . import nets
import torch.nn.functional as F

from . import utils


class SingleStep(torch.nn.Module):
    def __init__(self, obs_shape, action_dim, learning_rate, forward_model_weight):
        super().__init__()

        self.encoder = nets.Encoder(obs_shape).cuda()
        self.embed_dim = self.encoder.output_dim
        self.forward_model = nets.ForwardDynamics(self.embed_dim, action_dim).cuda()
        self.inverse_model = nets.InverseDynamics(self.embed_dim, action_dim).cuda()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters()),
            lr=learning_rate,
            # weight_decay=h["weight_decay"],
        )

        self.forward_model_weight = forward_model_weight

    def train_step(self, batch):
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")

        o_encoded = self.encoder(obs)
        on_encoded = self.encoder(obs_next)
        forward_model_loss = F.mse_loss(
            self.forward_model(o_encoded, act),
            on_encoded,
        )
        # print(torch.mean(o_encoded))
        # print(on_encoded)
        # print(self.forward_model(o_encoded, act))
        inverse_model_pred = self.inverse_model(o_encoded, on_encoded)
        inverse_model_loss = F.cross_entropy(
            inverse_model_pred,
            act,
        )
        # print((torch.argmax(inverse_model_pred, dim=1).cpu() == torch.arange(8).unsqueeze(1)).float().mean(dim=1))
        # print(act[0])
        # import cv2
        # cv2.imshow("", ((obs[0, :3].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8).repeat(10, axis=0).repeat(10, axis=1))
        # cv2.waitKey(0)
        # cv2.imshow("", ((obs[0, 3:].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8).repeat(10, axis=0).repeat(10, axis=1))
        # cv2.waitKey(0)
        # cv2.imshow("", ((obs_next[0, :3].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8).repeat(10, axis=0).repeat(10, axis=1))
        # cv2.waitKey(0)
        # cv2.imshow("", ((obs_next[0, 3:].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8).repeat(10, axis=0).repeat(10, axis=1))
        # cv2.waitKey(0)
        total_loss = self.forward_model_weight * forward_model_loss + inverse_model_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "forward": forward_model_loss.detach().item(),
            "inverse": inverse_model_loss.detach().item(),
            "total": total_loss.detach().item(),
            "accuracy": torch.mean(
                (torch.argmax(inverse_model_pred, dim=1) == act).float()
            )
            .detach()
            .item(),
        }


class MultiStep(torch.nn.Module):
    def __init__(
        self, obs_shape, action_dim, ss_encoder, learning_rate, gamma, tau, sync_freq
    ):
        super().__init__()
        self.encoder = nets.Encoder(obs_shape).cuda()

        self.action_dim = action_dim
        self.forward_model = nets.ForwardDynamics(
            self.encoder.output_dim, self.action_dim
        ).cuda()

        self.encoder_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()),
            lr=learning_rate,
        )
        self.forward_model_optimizer = torch.optim.Adam(
            self.forward_model.parameters(),
            lr=learning_rate,
        )

        self.target_encoder = deepcopy(self.encoder).cuda()

        self.gamma = gamma
        self.tau = tau
        # self.forward_loss_weight = h["forward_loss_weight"]
        self.ss_encoder = ss_encoder
        self.sync_freq = sync_freq
        # self.mico_beta = h["mico_beta"]

        self.steps_until_sync = self.sync_freq

    def train_step(self, batch):
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")

        o_encoded_online = self.encoder(obs)
        with torch.no_grad():
            o_encoded_target = self.target_encoder(obs)
            on_encoded_target = self.target_encoder(obs_next)

        self.forward_model_optimizer.zero_grad()
        forward_model_loss = F.mse_loss(
            self.forward_model(o_encoded_target, act),
            on_encoded_target,
        )
        forward_model_loss.backward()
        self.forward_model_optimizer.step()

        with torch.no_grad():
            ss_encoded = self.ss_encoder(obs)
            ss_distances = utils.pairwise_l1_distance(ss_encoded.detach()) * (
                1 - self.gamma
            )

            pred_on_encoded = self.forward_model(
                o_encoded_target.unsqueeze(1).expand(-1, self.action_dim, -1),
                torch.arange(self.action_dim, device="cuda")
                .unsqueeze(0)
                .expand(o_encoded_target.shape[0], -1),
            )  # shape (n, 4, e)

            target_distances = utils.pairwise_l1_distance(
                pred_on_encoded
            )  # shape (n**2, 4)
            target_distances = torch.mean(target_distances, dim=-1)

        distances = utils.pairwise_l1_distance(o_encoded_online)
        ms_loss = F.smooth_l1_loss(
            distances, ss_distances + self.gamma * target_distances
        )
        self.encoder_optimizer.zero_grad()
        ms_loss.backward()
        self.encoder_optimizer.step()

        if self.steps_until_sync == 0:
            self._sync_params()
            self.steps_until_sync = self.sync_freq
        else:
            self.steps_until_sync -= 1

        return {
            "ms_loss": ms_loss.detach().item(),
            "forward_loss": forward_model_loss.detach().item(),
        }

    def _sync_params(self):
        for curr, targ in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            targ.data.copy_(targ.data * (1.0 - self.tau) + curr.data * self.tau)


class SingleStepVAE(torch.nn.Module):
    def __init__(self, obs_shape, action_dim, h):
        super().__init__()

        self.encoder = nets.StochasticEncoder(obs_shape, h["latent_dim"]).cuda()
        # self.det_encoder = nets.DeterministicEncoder(obs_shape, h["latent_dim"]).cuda()
        self.embed_dim = h["latent_dim"]
        # self.forward_model = nets.ForwardDynamics(self.embed_dim, action_dim).cuda()
        self.inverse_model = nets.InverseDynamics(self.embed_dim, action_dim).cuda()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            # + list(self.det_encoder.parameters())
            # + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters()),
            lr=h["learning_rate"],
        )
        # self.kl_penalty_weight = h["kl_penalty_weight"]
        self.forward_model_weight = h["forward_model_weight"]
        self.starting_beta = 0
        self.ending_beta = 0.35
        self.annealing_start = 5000
        self.annealing_end = 50000
        # self.cycle_length = 60000
        # self.zero_proportion = 0.1

        self.step_num = 0
        self.beta = h["kl_penalty_weight"]
        # self.log_beta = torch.zeros(1, requires_grad=True, device="cuda")
        # self.beta_optim = torch.optim.Adam([self.log_beta], lr=h["beta_lr"])
        # self.kl_target = np.log(np.prod(action_dim))
        # self.beta = self.log_beta.detach().exp()

    def train_step(self, batch):
        # if self.annealing_start < self.step_num <= self.annealing_end:
        #     self.beta = self.starting_beta + (self.ending_beta - self.starting_beta) * (self.step_num - self.annealing_start) / (self.annealing_end - self.annealing_start)
        # step_offset = (self.step_num % self.cycle_length) / self.cycle_length
        # if step_offset <= self.zero_proportion:
        #     self.beta = 0
        # else:
        #     self.beta = self.starting_beta + (self.ending_beta - self.starting_beta) * (step_offset - self.zero_proportion) / (1 - self.zero_proportion)
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")

        o_mu, o_log_var = self.encoder.mu_log_var(obs)
        # o_encoded = self.det_encoder(obs)
        on_mu, on_log_var = self.encoder.mu_log_var(obs_next)
        # o_encoded = utils.reparameterize(o_mu, o_log_var)
        on_encoded = utils.reparameterize(on_mu, on_log_var)
        # forward_mu, forward_log_var = self.forward_model(o_mu.detach(), act)
        # forward_model_loss = torch.mean(
        #     0.5 * (on_encoded - forward_mu)**2 / forward_log_var.exp()
        #     + 0.5 * forward_log_var
        # )
        # forward_model_loss = F.mse_loss(
        #     self.forward_model(o_mu.detach(), act),
        #     on_encoded,
        # )
        inverse_model_loss = F.cross_entropy(
            self.inverse_model(o_mu.detach(), on_encoded),
            act,
        )
        # o_kl_loss = torch.mean(-0.5 * torch.sum(1 + o_log_var - o_mu**2 - o_log_var.exp(), dim=1), dim=0)
        on_kl_loss_batch = -0.5 * torch.sum(
            1 + on_log_var - on_mu**2 - on_log_var.exp(), dim=1
        )
        on_kl_loss = torch.mean(on_kl_loss_batch)
        total_loss = inverse_model_loss + self.beta * on_kl_loss
        # total_loss = self.forward_model_weight * forward_model_loss \
        #     + (1 - self.forward_model_weight) * inverse_model_loss \
        #     + self.beta * on_kl_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # self.beta_optim.zero_grad()
        # beta_loss = torch.mean((self.kl_target - on_kl_loss_batch.detach()) * self.log_beta.exp())
        # beta_loss.backward()
        # self.beta_optim.step()
        # self.beta = self.log_beta.detach().exp()

        self.step_num += 1
        return {
            "inverse": inverse_model_loss.detach().item(),
            # "o_kl_loss": o_kl_loss.detach().item(),
            "on_kl_loss": on_kl_loss.detach().item(),
            # "beta_loss": beta_loss.detach().item(),
            # "forward": forward_model_loss.detach().item(),
            "total": total_loss.detach().item(),
            "beta": self.beta,
        }


class AdversarialSingleStep(torch.nn.Module):
    def __init__(self, obs_shape, action_dim, h):
        super().__init__()

        self.encoder = nets.Encoder(obs_shape).cuda()
        self.embed_dim = self.encoder.output_dim
        self.active_model = nets.ForwardDynamics(self.embed_dim, action_dim).cuda()
        self.passive_model = nets.PassiveForwardDynamics(self.embed_dim)
        self.encoder_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()), lr=h["encoder_learning_rate"]
        )
        self.active_optimizer = torch.optim.Adam(
            list(self.active_model.parameters()),
            lr=h["active_learning_rate"],
        )
        self.passive_optimizer = torch.optim.Adam(
            list(self.passive_model.parameters()), lr=h["passive_learning_rate"]
        )

        self.adversarial_loss_weight = h["adversarial_loss_weight"]
        self.num_steps = 0

    def train_step(self, batch):
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")

        o_encoded = self.encoder(obs)
        on_encoded = self.encoder(obs_next)

        # train passive model
        passive_loss = F.smooth_l1_loss(
            self.passive_model(o_encoded.detach()),
            on_encoded.detach(),
        )
        self.passive_optimizer.zero_grad()
        passive_loss.backward()
        self.passive_optimizer.step()
        self.passive_optimizer.zero_grad()

        # train active model
        active_loss = F.smooth_l1_loss(
            self.active_model(o_encoded.detach(), act),
            on_encoded.detach(),
        )
        self.active_optimizer.zero_grad()
        active_loss.backward()
        self.active_optimizer.step()
        self.active_optimizer.zero_grad()

        ret = {
            "active": active_loss.detach().item(),
            "passive": passive_loss.detach().item(),
        }

        # train encoder
        if self.num_steps % 5 == 0:
            passive_loss_adv = F.mse_loss(
                self.passive_model(o_encoded),
                on_encoded,
            )
            active_loss_adv = F.mse_loss(
                self.active_model(o_encoded, act),
                on_encoded,
            )
            adversarial_loss = (
                self.adversarial_loss_weight * -torch.log(passive_loss_adv)
                + (1 - self.adversarial_loss_weight) * active_loss_adv
            )
            self.encoder_optimizer.zero_grad()
            adversarial_loss.backward()
            self.encoder_optimizer.step()
            self.encoder_optimizer.zero_grad()

            ret["adversarial"] = adversarial_loss.detach().item()

        self.num_steps += 1
        return ret


class BehavioralCloning(torch.nn.Module):
    def __init__(self, obs_shape, action_dim, h):
        super().__init__()

        self.encoder = nets.DeterministicEncoder(obs_shape, h["latent_dim"]).cuda()
        self.embed_dim = self.encoder.output_dim
        self.dqn = nets.DQN(self.embed_dim, action_dim).cuda()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.dqn.parameters()),
            lr=h["learning_rate"],
        )

    def train_step(self, batch):
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")

        o_encoded = self.encoder(obs)
        loss = F.cross_entropy(
            self.dqn(o_encoded),
            act,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.detach().item()}

    def evaluate(self, batch):
        with torch.no_grad():
            obs = torch.as_tensor(batch["obs"], device="cuda")
            act = torch.as_tensor(batch["action"], device="cuda")

            o_encoded = self.encoder(obs)
            loss = F.cross_entropy(
                self.dqn(o_encoded),
                act,
            )
        return loss.detach().item()

    def forward(self, x):
        return self.dqn(self.encoder(x))


class Autoencoder(torch.nn.Module):
    def __init__(self, obs_shape, action_dim, h):
        super().__init__()

        self.encoder = nets.StochasticEncoder(obs_shape, h["latent_dim"]).cuda()
        self.embed_dim = self.encoder.output_dim
        self.decoder = nets.Decoder(self.embed_dim, obs_shape)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=h["learning_rate"],
        )

        self.starting_beta = 0
        self.ending_beta = 0.35
        self.annealing_start = 10000
        self.annealing_end = 100000

        self.step_num = 0
        self.beta = self.starting_beta

    def train_step(self, batch):
        obs = torch.as_tensor(batch["obs"], device="cuda")

        if self.annealing_start < self.step_num <= self.annealing_end:
            self.beta = self.starting_beta + (self.ending_beta - self.starting_beta) * (
                self.step_num - self.annealing_start
            ) / (self.annealing_end - self.annealing_start)

        mu, log_var = self.encoder.mu_log_var(obs)
        obs_recon = self.decoder(utils.reparameterize(mu, log_var))
        recon_loss = F.mse_loss(obs_recon, obs, reduction="sum") / obs.shape[0]
        kl_loss_batch = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        kl_loss = torch.mean(kl_loss_batch)

        total_loss = recon_loss + self.beta * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.step_num += 1

        return {
            "total_loss": total_loss.detach().item(),
            "recon_loss": recon_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
            "beta": self.beta,
        }, obs_recon.detach()
