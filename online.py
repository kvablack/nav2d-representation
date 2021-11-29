import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque
from copy import deepcopy
from multiprocessing import Pool

import h5py
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np

import nets
import utils
from dqn_her import DQN_HER

np.set_printoptions(threshold=10000)
import random
from nav2d import Navigate2D


def train(h, args):
    # dataset is used only to get environment hyperparameters
    dataset = h5py.File(args.dataset, "r")
    d = dict(dataset.attrs)
    # d["num_obstacles"] = 7
    env = Navigate2D(d)

    if args.model_path is not None:
        model = torch.load(args.model_path)
        encoder = model.encoder
        dqn = getattr(model, "dqn", None)
    else:
        encoder = nets.DeterministicEncoder(env.observation_space.shape, 32)
        dqn = None

    assert dqn is None

    log_path = args.logdir
    if os.path.exists(log_path):
        response = input("Overwrite [y/n]? ")
        if response == "n":
            sys.exit(1)
        elif response == "y":
            shutil.rmtree(log_path)
        else:
            raise RuntimeError()
    os.makedirs(log_path)

    with open(os.path.join(log_path, "hyperparameters.json"), "w") as f:
        json.dump(dict(vars(args), **h), f)

    tensorboard = SummaryWriter(log_path)

    random.seed(h["seed"])
    torch.manual_seed(h["seed"])
    env.seed(h["seed"])

    dqn_her = DQN_HER(env, encoder, h, dqn)

    successes = deque(maxlen=100)
    global_step = 0
    steps_until_save = 0
    while global_step < h["steps"]:
        total_reward, avg_loss, min_dist, ep_len = dqn_her.run_episode()
        global_step += ep_len
        steps_until_save += ep_len
        successes.append(min_dist == 0)
        success_rate = np.mean(successes)
        tensorboard.add_scalar("ep_rew", total_reward, global_step)
        tensorboard.add_scalar("ep_loss", avg_loss, global_step)
        tensorboard.add_scalar("min_dist", min_dist, global_step)
        tensorboard.add_scalar("success_rate", success_rate, global_step)

        if steps_until_save >= 1000:
            torch.save(dqn_her.model, os.path.join(log_path, f"model_{global_step}.pt"))
            steps_until_save = 0

    torch.save(dqn_her.model, os.path.join(log_path, "model_final.pt"))


if __name__ == "__main__":
    HYPERPARAMETERS = {
        "batch_size": 16,
        "seed": 1,
        "n_epochs": 50,
        "epsilon_high": 0.9,
        "epsilon_low": 0.2,
        "epsilon_decay": 10000,
        "steps": 100000,
        "gamma": 0.99,
        "replay_buffer_size": 500000,
        "use_ddqn": True,
        "learning_rate": 0.0001,
        "tau": 0.005,
    }

    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    train_args = []
    for i in range(5):
        h = deepcopy(HYPERPARAMETERS)
        h["seed"] = i
        a = deepcopy(args)
        a.logdir = os.path.join(args.logdir, str(i))
        train_args.append((h, a))

    with Pool(5) as p:
        p.starmap(train, train_args)
