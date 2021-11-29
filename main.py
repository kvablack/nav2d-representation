import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque

import h5py
import tqdm
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter

import nets
import torch
import numpy as np

import utils
from models import SingleStep, SingleStepVAE, MultiStep

np.set_printoptions(threshold=10000)
import torch.nn.functional as F
import random
from nav2d import Navigate2D


def train(h, args):
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

    dataset = h5py.File(args.dataset, "r")
    dataset_keys = []
    dataset.visit(lambda key: dataset_keys.append(key) if isinstance(dataset[key], h5py.Dataset) else None)
    env = Navigate2D(dataset.attrs)

    ss_model = SingleStep(env.observation_space.shape, env.action_space.n, h["single_step"]).cuda()
    # ss_model = torch.load("experiments/ss_pretrain/single_step.pt")
    ms_model = MultiStep(env.observation_space.shape, ss_model.encoder, env, h["multi_step"]).cuda()

    global_step = 0
    for epoch in range(h["n_epochs"]):
        sample_ind_all = np.random.permutation(len(dataset["obs"]))
        for i in tqdm.tqdm(range(-(len(sample_ind_all) // -h["batch_size"])), desc=f"Epoch #{epoch}"):
            start = i * h["batch_size"]
            end = min(len(sample_ind_all), (i + 1) * h["batch_size"])
            sample_ind = np.sort(sample_ind_all[start:end])
            samples = {key: dataset[key][sample_ind] for key in dataset_keys}

            ss_losses = ss_model.train_step(samples)
            tensorboard.add_scalars("ss", ss_losses, global_step)

            ms_losses = ms_model.train_step(samples)
            tensorboard.add_scalar("ms/loss", ms_losses["loss"], global_step)
            # tensorboard.add_scalar("ms/mean_norm", ms_losses["mean_norm"], global_step)
            # tensorboard.add_scalar("ms/mean_ang", ms_losses["mean_ang"], global_step)

            if global_step > 0 and global_step % 1000 == 0:
                ss_heatmaps = utils.perturb_heatmap(samples["obs"][-1], ss_model.encoder)
                tensorboard.add_images("single_step", np.stack(ss_heatmaps), global_step)

                ms_heatmaps = utils.perturb_heatmap(samples["obs"][-1], ms_model.encoder)
                tensorboard.add_images("multi_step", np.stack(ms_heatmaps), global_step)

                torch.save(ss_model, os.path.join(log_path, f"single_step_{global_step}.pt"))
                torch.save(ms_model, os.path.join(log_path, f"multi_step_{global_step}.pt"))

            global_step += 1


if __name__ == "__main__":
    HYPERPARAMETERS = {
        "single_step": {
            "forward_model_weight": 0.8,
            "learning_rate": 0.001,
            # "latent_dim": 32,
            # "weight_decay": 0.0,
        },
        "multi_step": {
            "tau": 0.005,
            "sync_freq": 1,
            "gamma": 0.99,
            "learning_rate": 0.0001,
        },
        "batch_size": 128,
        "seed": 1,
        "n_epochs": 40,
    }

    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    train(HYPERPARAMETERS, args)
