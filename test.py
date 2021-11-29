import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import cm
import os
import shutil

from nav2d import Navigate2D
import utils

env = Navigate2D(
    {
        "grid_size": 20,
        "scale": 1,
        "num_obstacles": 15,
        "obstacle_diameter": 1,
        "min_goal_dist": 10,
        "use_factorized_state": False,
        "max_episode_length": 50,
    }
)
env.seed(0)

path = "experiments/vae/beta_0_1_10000/"
ss_model = torch.load(path + "single_step.pt")
obs = env.reset()
encoded = ss_model.encoder.mu_log_var(
    torch.as_tensor(obs, device="cuda").unsqueeze(0)
)
print(encoded)
sys.exit(0)

ms_model = torch.load(path + "multi_step.pt")
if os.path.exists(path + "projector"):
    shutil.rmtree(path + "projector")
tensorboard = SummaryWriter(path + "projector")


def encode(obs):
    return ms_model.encoder(
        torch.as_tensor(obs, device="cuda").unsqueeze(0)
    ).squeeze(0).detach().cpu().numpy()


def render(obs):
    return utils.render(obs).transpose([1, 2, 0])


embeddings = []
labels = []
images = []

mid = env.size // 2
pos = [mid + 1, mid]

obs = env.reset()
obs[0, mid:mid + 2, :] = -1
obs[1, :, :] = -1
obs[1, pos[0], pos[1]] = 1
obs_encoded = encode(obs)
plt.imshow(render(obs))
plt.title("original")
plt.savefig("render/original.png")
embeddings.append(obs_encoded)
labels.append("original")
images.append(utils.render(obs))

heatmap = utils.perturb_heatmap(obs, ms_model.encoder)[1]
plt.imshow(heatmap.transpose([1, 2, 0]))
# plt.show()

for i in range(0, mid):
    obs_1block = obs.copy()
    obs_1block[0, mid, i] = 1
    plt.imshow(render(obs_1block))
    plt.title(np.linalg.norm(obs_encoded - encode(obs_1block), ord=1))
    plt.savefig(f"render/{i}_1block.png")
    embeddings.append(encode(obs_1block))
    labels.append(f"{i}_1block")
    images.append(utils.render(obs_1block))

    obs_2block = obs_1block.copy()
    obs_2block[0, mid + 1, i] = 1
    plt.imshow(render(obs_2block))
    plt.title(np.linalg.norm(obs_encoded - encode(obs_2block), ord=1))
    plt.savefig(f"render/{i}_2block.png")
    embeddings.append(encode(obs_2block))
    labels.append(f"{i}_2block")
    images.append(utils.render(obs_2block))

obs_4block = obs_2block.copy()
obs_4block[0, mid:mid + 2, mid - 2] = 1
plt.imshow(render(obs_4block))
plt.title(np.linalg.norm(obs_encoded - encode(obs_4block), ord=1))
plt.savefig(f"render/4block.png")
embeddings.append(encode(obs_4block))
labels.append(f"4block")
images.append(utils.render(obs_4block))

obs_moved = obs.copy()
obs_moved[1, pos[0], pos[1]] = -1
obs_moved[1, pos[0], pos[1] + 1] = 1
plt.imshow(render(obs_moved))
plt.title(np.linalg.norm(obs_encoded - encode(obs_moved), ord=1))
plt.savefig("render/moved.png")
embeddings.append(encode(obs_moved))
labels.append(f"moved")
images.append(utils.render(obs_moved))

obs_same = env.reset()
obs_same[0, mid:mid + 2, :] = -1
obs_same[1, :, :] = -1
obs_same[1, pos[0], pos[1]] = 1
plt.imshow(render(obs_same))
plt.title(np.linalg.norm(obs_encoded - encode(obs_same), ord=1))
plt.savefig("render/same.png")
embeddings.append(encode(obs_same))
labels.append(f"same")
images.append(utils.render(obs_same))

tensorboard.add_embedding(np.array(embeddings), labels, torch.as_tensor(images))

# pos = obs[1, :, :] == 1
# c, h, w = obs.shape
# encoded = ms_model.encoder(
#     torch.as_tensor(obs, device="cuda").unsqueeze(0)
# ) \
#     .squeeze(0).detach().cpu().numpy()
#
# obs_perturbed = np.broadcast_to(obs, [h * w, 3, h, w]).copy()
# mask = (-np.eye(h * w) * 2 + 1).reshape(h * w, h, w)
# obs_perturbed[:, 1] *= mask
# obs_perturbed[:, 1, pos] = -1
# encoded_perturbed = ms_model.encoder(
#     torch.as_tensor(obs_perturbed, device="cuda")
# ) \
#     .detach().cpu().numpy()
#
# distances = np.linalg.norm(encoded - encoded_perturbed, ord=1, axis=-1).reshape(h, w)
# distances /= np.max(distances)
# heatmap = cm.gray(distances)[:, :, :3]
# heatmap[pos, :] = [1, 0, 0]
# plt.imshow(heatmap)
# plt.show()