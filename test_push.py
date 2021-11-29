import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import sys
np.set_printoptions(suppress=True)

from torch.utils.tensorboard import SummaryWriter

from nav2d_push import Navigate2DPush
import utils


env = Navigate2DPush(
    {
        "grid_size": 20,
        "block_diameter": 2,
        "min_goal_dist": 10,
        "use_factorized_state": False,
        "max_episode_length": 50,
    }
)
env.seed(0)


path = "experiments/push/vae/beta_0/"
# path = "experiments/push/vae/multistep_from_frozen/"
ss_model = torch.load(path + "single_step_499000.pt")
if os.path.exists(path + "projector"):
    shutil.rmtree(path + "projector")
tensorboard = SummaryWriter(path + "projector")

def encode(obs):
    return ss_model.encoder(
        torch.as_tensor(obs, device="cuda").unsqueeze(0)
    ).squeeze(0).detach().cpu().numpy()

def render(obs):
    return utils.render(obs).transpose([1, 2, 0])

def perturb_heatmap_push(obs):
    heatmap = np.empty([obs.shape[1], obs.shape[2]], dtype=np.float32)
    player_pos = np.argwhere(obs[1] == 1)[0]
    player_block_ind = np.array(np.broadcast_arrays(*env._block_ind(player_pos))).reshape(2, -1).T
    obs_encoded = encode(obs)
    for y in range(heatmap.shape[0]):
        for x in range(heatmap.shape[1]):
            block_ind = env._block_ind((y, x))
            if (np.array([y, x]) == player_block_ind).all(axis=1).any():
                heatmap[y, x] = 0
                continue
            obs_perturb = obs.copy()
            obs_perturb[0, :, :] = -1
            obs_perturb[0][block_ind] = 1
            heatmap[y, x] = np.linalg.norm(obs_encoded - encode(obs_perturb), ord=1)
    return cm.gray(heatmap)[:, :, :3] \
        .repeat(2, axis=0).repeat(2, axis=1)

while True:
    obs = env.reset()
    img, heatmap = utils.perturb_heatmap_push(obs, ss_model.encoder, env)
    plt.imshow(img.T)
    plt.show()
    plt.imshow(heatmap.T)
    plt.show()

embeddings = []
labels = []
images = []
def add(x, y, label):
    obs = np.empty([3, 20, 20], dtype=np.float32)
    obs[:, :, :] = -1
    obs[1, 10, 10] = 1
    obs[0][env._block_ind((x, y))] = 1
    embeddings.append(encode(obs))
    labels.append(label)
    images.append(utils.render(obs))
    # plt.imshow(render(obs))
    # plt.title(label)
    # plt.show()

def add(x, y):
    obs = np.empty([3, 20, 20], dtype=np.float32)
    obs[:, :, :] = -1
    obs[1, x, y] = 1
    obs[0][env._block_ind((10, 10))] = 1
    print(x, y, encode(obs))


# obs = np.empty([3, 20, 20], dtype=np.float32)
# obs[:, :, :] = -1
# obs[1, 10, 10] = 1
# plt.imshow(perturb_heatmap_push(obs))
# plt.show()
# sys.exit(0)

add(0, 0)
add(0, 1)
add(0, 2)
add(0, 3)
add(1, 0)
add(2, 0)
add(3, 0)
sys.exit(0)

add(7, 9, "topl_0")
add(7, 10, "topm_0")
add(7, 11, "topr_0")
add(6, 9, "topl_1")
add(6, 10, "topm_1")
add(6, 11, "topr_1")
add(5, 9, "topl_2")
add(5, 10, "topm_2")
add(5, 11, "topr_2")
add(4, 9, "topl_3")
add(4, 10, "topm_3")
add(4, 11, "topr_3")

add(9, 7, "leftt_0")
add(10, 7, "leftm_0")
add(11, 7, "leftb_0")
add(9, 6, "leftt_1")
add(10, 6, "leftm_1")
add(11, 6, "leftb_1")
add(9, 5, "leftt_2")
add(10, 5, "leftm_2")
add(11, 5, "leftb_2")
add(9, 4, "leftt_3")
add(10, 4, "leftm_3")
add(11, 4, "leftb_3")

tensorboard.add_embedding(np.array(embeddings), labels, torch.as_tensor(images))

# obs, rew, done, info = env.step(1)
# obs, rew, done, info = env.step(1)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs, rew, done, info = env.step(0)
# obs0 = obs.copy()
# obs, rew, done, info = env.step(0)


# obs_next = env.forward_oracle(np.array([obs]))
# for i in range(4):
#     plt.imshow(render(obs_next[i, 0]))
#     plt.show()
