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
from models import SingleStep, SingleStepVAE, MultiStep, BehavioralCloning

np.set_printoptions(threshold=10000)
import torch.nn.functional as F
import random
from nav2d.nav2d import Navigate2D


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

    bc_model = BehavioralCloning(env.observation_space.shape, env.action_space.n, h).cuda()

    num_train = int(len(dataset["obs"]) * 0.8)
    num_val = len(dataset["obs"]) - num_train

    global_step = 0
    for epoch in range(h["n_epochs"]):
        sample_ind_all = np.random.permutation(num_train)
        for i in tqdm.tqdm(range(-(len(sample_ind_all) // -h["batch_size"])), desc=f"Epoch #{epoch}"):
            start = i * h["batch_size"]
            end = min(len(sample_ind_all), (i + 1) * h["batch_size"])
            sample_ind = np.sort(sample_ind_all[start:end])
            samples = {key: dataset[key][sample_ind] for key in dataset_keys}

            loss = bc_model.train_step(samples)
            tensorboard.add_scalars("bc", loss, global_step)

            if global_step > 0 and global_step % 1000 == 0:
                heatmaps = utils.perturb_heatmap(samples["obs"][-1], bc_model.encoder)
                tensorboard.add_images("bc", np.stack(heatmaps), global_step)

                bc_model.eval()
                val_loss = 0
                num_val_batch = -(num_val // -h["val_batch_size"])
                for j in range(num_val_batch):
                    start = j * h["val_batch_size"]
                    end = min(num_val, (j + 1) * h["val_batch_size"])
                    val_samples = {key: dataset[key][num_train + start:num_train + end] for key in dataset_keys}
                    val_loss += bc_model.evaluate(val_samples)
                tensorboard.add_scalars("bc", {"val_loss": val_loss / num_val_batch}, global_step)
                bc_model.train()

                torch.save(bc_model, os.path.join(log_path, "model.pt"))

            global_step += 1

    torch.save(bc_model, os.path.join(log_path, "model.pt"))


if __name__ == "__main__":
    HYPERPARAMETERS = {
        "batch_size": 128,
        "val_batch_size": 1024,
        "seed": 1,
        "n_epochs": 6,
        "learning_rate": 0.001,
        "latent_dim": 32,
    }

    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    train(HYPERPARAMETERS, args)
