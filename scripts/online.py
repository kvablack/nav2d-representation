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

from nav2d_representation import nets
from nav2d_representation.dqn_her import DQN_HER
from nav2d_representation.utils import ENV_DICT

np.set_printoptions(threshold=10000)
import random


def train(args):
    # dataset is used only to get environment hyperparameters
    dataset = h5py.File(args.dataset, "r")
    attrs = dict(dataset.attrs)
    # attrs["num_obstacles"] = 7
    env_name = attrs.pop("env")
    env = ENV_DICT[env_name](**attrs)

    if args.model_path is not None:
        model = torch.load(args.model_path)
        encoder = model.encoder
        dqn = getattr(model, "dqn", None)
    else:
        encoder = nets.Encoder(env.observation_space.shape)
        dqn = None

    tensorboard = SummaryWriter(log_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    dqn_her = DQN_HER(
        env,
        encoder,
        dqn=dqn,
        gamma=args.gamma,
        tau=args.tau,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epsilon_low=args.epsilon_low,
        epsilon_high=args.epsilon_high,
        epsilon_decay=args.epsilon_decay,
        updates_per_step=args.updates_per_step,
        replay_buffer_size=args.replay_buffer_size,
    )

    successes = deque(maxlen=100)
    global_step = 0
    steps_until_save = 0
    while global_step < args.total_steps:
        total_reward, avg_loss, min_dist, ep_len = dqn_her.run_episode()
        global_step += ep_len
        steps_until_save += ep_len
        successes.append(min_dist == 0)
        success_rate = np.mean(successes)
        tensorboard.add_scalar("ep_rew", total_reward, global_step)
        tensorboard.add_scalar("ep_loss", avg_loss, global_step)
        tensorboard.add_scalar("min_dist", min_dist, global_step)
        tensorboard.add_scalar("success_rate", success_rate, global_step)
        # tensorboard.add_scalar(
        #     "alpha", torch.sigmoid(dqn_her.model[0].alpha.cpu().detach()), global_step
        # )

        if steps_until_save >= 10000:
            torch.save(dqn_her.model, os.path.join(log_path, f"model_{global_step}.pt"))
            steps_until_save = 0

    torch.save(dqn_her.model, os.path.join(log_path, "model_final.pt"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon_high", type=float, default=0.9)
    parser.add_argument("--epsilon_low", type=float, default=0.2)
    parser.add_argument("--epsilon_decay", type=int, default=10000)
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--replay_buffer_size", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--updates_per_step", type=int, default=1)

    parser.add_argument("--model-path", type=str, required=False)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_seeds", type=int, required=True)
    args = parser.parse_args()

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

    with open(os.path.join(log_path, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    train_args = []
    for i in range(args.num_seeds):
        a = deepcopy(args)
        a.logdir = os.path.join(log_path, str(i))
        a.seed = args.seed + i
        train_args.append(a)

    with Pool(args.num_seeds) as p:
        p.map(train, train_args)
