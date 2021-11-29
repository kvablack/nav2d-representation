import json
import os
import shutil
import sys
from collections import deque
from copy import deepcopy
from multiprocessing import Pool

from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter

import nets
import torch
import numpy as np

import utils
from models import SingleStep, MultiStep, SingleStepVAE

np.set_printoptions(threshold=10000)
import torch.nn.functional as F
import random
from nav2d_push import Navigate2DPush


def train(h):
    log_path = h["log_path"]
    if os.path.exists(log_path):
        print(f"Directory {log_path} exists")
        sys.exit(1)
    os.makedirs(log_path)

    with open(os.path.join(log_path, "hyperparameters.json"), "w") as f:
        json.dump(h, f)

    tensorboard = SummaryWriter(log_path)

    env = Navigate2DPush(h["environment"])
    random.seed(h["seed"])
    torch.manual_seed(h["seed"])
    env.seed(h["seed"])

    ss_model = SingleStep(env.observation_space.shape, env.action_space.n, h["single_step"]).cuda()
    # ss_model = torch.load("experiments/push/vae/beta_0/single_step_499000.pt")
    # ms_model = MultiStep(env.observation_space.shape, ss_model.encoder, env, h["multi_step"]).cuda()

    replay_buffer = deque(maxlen=h["replay_buffer_size"])
    batch_size = h["batch_size"]

    global_step = 0
    for episode in range(h["num_episodes"]):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs_next, rew, done, info = env.step(action)
            replay_buffer.append([obs, action, rew, obs_next, done, info])
            obs = obs_next

            if len(replay_buffer) >= batch_size:
                samples = random.sample(replay_buffer, batch_size)

                ss_losses = ss_model.train_step(samples)
                tensorboard.add_scalars("ss", ss_losses, global_step)

                # ms_losses = ms_model.train_step(samples)
                # tensorboard.add_scalar("ms/loss", ms_losses["loss"], global_step)
                # tensorboard.add_scalar("ms/mean_norm", ms_losses["mean_norm"], global_step)
                # tensorboard.add_scalar("ms/mean_ang", ms_losses["mean_ang"], global_step)
                global_step += 1

            if global_step > 0 and global_step % 1000 == 0:
                ss_heatmaps = utils.perturb_heatmap_push(replay_buffer[-1][0], ss_model.encoder, env)
                tensorboard.add_images("single_step", np.stack(ss_heatmaps), global_step)

                # ms_heatmaps = utils.perturb_heatmap_push(replay_buffer[-1][0], ms_model.encoder, env)
                # tensorboard.add_images("multi_step", np.stack(ms_heatmaps), global_step)

            if global_step > 0 and global_step % 1000 == 0:
                torch.save(ss_model, os.path.join(log_path, f"single_step_{global_step}.pt"))
                # torch.save(ms_model, os.path.join(log_path, f"multi_step_{global_step}.pt"))


if __name__ == "__main__":
    HYPERPARAMETERS = {
        "environment": {
            "grid_size": 20,
            "block_diameter": 2,
            "min_goal_dist": 10,
            "use_factorized_state": False,
            "max_episode_length": 50,
        },
        "single_step": {
            # "kl_penalty_weight": 0,
            "learning_rate": 0.001,
            "weight_decay": 0.00005,
            # "forward_model_weight": 0.2,
            "latent_dim": 32,
            # "beta_lr": 0.001,
        },
        "multi_step": {
            "tau": 0.005,
            "sync_freq": 1,
            "mico_beta": 0.1,
            "gamma": 0.99,
            "learning_rate": 0.0001,
        },
        "replay_buffer_size": 250000,
        "num_episodes": 10000,
        "batch_size": 256,
        "seed": 1,
    }

    train(HYPERPARAMETERS)
    # hyps = []
    # for beta in [0, 0.01, 0.001]:
    #     h = deepcopy(HYPERPARAMETERS)
    #     h["single_step"]["kl_penalty_weight"] = beta
    #     h["log_path"] = f"experiments/push/vae/latent_dim_2/beta_{beta}"
    #     hyps.append(h)
    #
    # with Pool(len(hyps)) as p:
    #     p.map(train, hyps)

# if __name__ == "__main__":
#     HYPERPARAMETERS = {
#         "environment": {
#             "grid_size": 20,
#             "block_diameter": 2,
#             "min_goal_dist": 10,
#             "use_factorized_state": False,
#             "max_episode_length": 50,
#         },
#         "single_step": {
#             "active_learning_rate": 0.0002,
#             "passive_learning_rate": 0.0002,
#             "encoder_learning_rate": 0.0002,
#             "adversarial_loss_weight": 0.01,
#         },
#         "multi_step": {
#             "tau": 0.005,
#             "sync_freq": 1,
#             "mico_beta": 0.1,
#             "gamma": 0.99,
#             "learning_rate": 0.0001,
#         },
#         "replay_buffer_size": 250000,
#         "num_episodes": 10000,
#         "batch_size": 16,
#         "seed": 1,
#         "log_path": "experiments/adv/first"
#     }
#
#     train(HYPERPARAMETERS)
