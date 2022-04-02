import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque
from d4rl.pointmaze.maze_model import LARGE_OPEN

import h5py
import tqdm
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np

from nav2d_representation.models import SingleStep, MultiStep

np.set_printoptions(threshold=10000)
import torch.nn.functional as F
import random
from nav2d_representation.utils import ENV_DICT
from nav2d_representation.nav2d.utils import perturb_heatmap


def train(args):
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

    tensorboard = SummaryWriter(log_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = h5py.File(args.dataset, "r")
    dataset_keys = []
    dataset.visit(
        lambda key: dataset_keys.append(key)
        if isinstance(dataset[key], h5py.Dataset)
        else None
    )

    attrs = dict(dataset.attrs)
    env_name = attrs.pop("env")
    env = ENV_DICT[env_name](**attrs)
    action_dim = env.action_space.n
    obs_shape = dataset["obs"][0].shape
    # obs_shape = (obs_shape[0] * obs_shape[1], obs_shape[2], obs_shape[3])
    print("Observation shape: ", obs_shape)
    print("Action dim: ", action_dim)

    mem_dataset = {}
    for key in dataset_keys:
        mem_dataset[key] = dataset[key][:]
    dataset = mem_dataset

    ss_model = SingleStep(
        obs_shape,
        action_dim,
        learning_rate=args.single_step_lr,
        forward_model_weight=args.single_step_forward_weight,
    ).cuda()
    ms_model = MultiStep(
        obs_shape,
        action_dim,
        ss_model.encoder,
        learning_rate=args.multi_step_lr,
        gamma=args.multi_step_gamma,
        tau=args.multi_step_tau,
        sync_freq=args.multi_step_sync_freq,
    ).cuda()

    global_step = 0
    for epoch in range(args.n_epochs):
        sample_ind_all = np.random.permutation(len(dataset["obs"]))
        for i in tqdm.tqdm(
            range(-(len(sample_ind_all) // -args.batch_size)), desc=f"Epoch #{epoch}"
        ):
            start = i * args.batch_size
            end = min(len(sample_ind_all), (i + 1) * args.batch_size)
            sample_ind = np.sort(sample_ind_all[start:end])
            samples = {key: dataset[key][sample_ind] for key in dataset_keys}

            ss_losses = ss_model.train_step(samples)
            tensorboard.add_scalars("ss", ss_losses, global_step)

            ms_losses = ms_model.train_step(samples)
            tensorboard.add_scalars("ms", ms_losses, global_step)

            if global_step % 10000 == 0:
                torch.save(
                    ss_model, os.path.join(log_path, f"single_step_{global_step}.pt")
                )
                torch.save(
                    ms_model, os.path.join(log_path, f"multi_step_{global_step}.pt")
                )

                if env_name == "nav2d":
                    ss_heatmaps = perturb_heatmap(samples["obs"][-1], ss_model.encoder)
                    tensorboard.add_images(
                        "single_step", np.stack(ss_heatmaps), global_step
                    )

                    ms_heatmaps = perturb_heatmap(samples["obs"][-1], ms_model.encoder)
                    tensorboard.add_images(
                        "multi_step", np.stack(ms_heatmaps), global_step
                    )

            global_step += 1
    torch.save(ss_model, os.path.join(log_path, "single_step_final.pt"))
    torch.save(ms_model, os.path.join(log_path, "multi_step_final.pt"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--single_step_forward_weight", type=float, default=0.8)
    parser.add_argument("--single_step_lr", type=float, default=0.001)
    parser.add_argument("--multi_step_tau", type=float, default=0.005)
    parser.add_argument("--multi_step_sync_freq", type=int, default=1)
    parser.add_argument("--multi_step_gamma", type=float, default=0.99)
    parser.add_argument("--multi_step_lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=40)
    args = parser.parse_args()

    train(args)
