import sys
from d4rl.pointmaze.maze_model import LARGE_OPEN

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import cm
import os
import cv2
import shutil
from gym.wrappers import FrameStack

from tqdm import tqdm

from nav2d_representation.pointmass.d4rl_maze2d import VisualMazeEnv
from nav2d_representation.pointmass.utils import create_video
from d4rl.pointmaze import waypoint_controller
import matplotlib.pyplot as plt
import h5py

ss_model = torch.load("pointmass_experiments/open_e0.2_notarget/single_step_final.pt").cuda()
with h5py.File(
    f"pointmass_dataset_LARGE_OPEN_s1_e0.2_size100000.hdf5", "r"
) as f:
    obs = f["obs"][:]
    obs_next = f["obs_next"][:]
    actions = f["action"][:]
    for i in range(8):
        print(i, np.mean((actions == i).astype(int)))

    binsize = 0.5
    speed_discrete = np.round(np.linalg.norm(f["info/qvel"][:], axis=1) / binsize) * binsize
    grid, totals = np.unique(speed_discrete, return_counts=True)
    speed_mask = grid == speed_discrete[:, None]  # shape (n, numbins)
    correct = np.zeros_like(totals)
    # vel_discrete = np.round(f["info/qvel"][:] / binsize) * binsize  # shape (n, 2)
    # grid = np.mgrid[slice(-20, 20 + binsize, binsize), slice(-20, 20 + binsize, binsize)]  # shape (2, numbins, numbins)
    # vel_mask = (vel_discrete[..., None, None] == grid[None, ...]).all(axis=1)  # shape (n, numbins, numbins)
    # totals = vel_mask.sum(axis=0)  # shape (numbins, numbins)
    # correct = np.zeros_like(totals)

    for i in tqdm(range(-(100000 // -256))):
        start = i * 256
        end = min(100000, (i + 1) * 256)
        o = torch.as_tensor(obs[start:end], device="cuda").flatten(1, 2) / 127.5 - 1
        on = torch.as_tensor(obs_next[start:end], device="cuda").flatten(1, 2) / 127.5 - 1
        with torch.no_grad():
            pred = torch.argmax(ss_model.inverse_model(ss_model.encoder(o), ss_model.encoder(on)), dim=1).cpu().numpy()
        correct_mask = pred == actions[start:end]  # shape n
        print(np.mean(correct_mask))
        mask = np.logical_and(correct_mask[:, None], speed_mask[start:end])  # shape (n, numbins)
        correct += mask.sum(axis=0)
        # mask = np.logical_and(correct_mask[:, None, None], vel_mask[start:end])  # shape(n, numbins, numbins)
        # correct += mask.sum(axis=0)
    
    z = correct / totals
    std = np.sqrt(z * (1 - z) / totals)
    plt.plot(grid, z)
    plt.fill_between(grid, z - std * 2, z + std * 2, facecolor="tab:blue", alpha=0.5)
    plt.show()

    # z = np.where(totals != 0, correct / totals, 0)
    # z = z[:-1, :-1]

    # levels = MaxNLocator(nbins=100).tick_values(z.min(), z.max())
    # cmap = plt.get_cmap('Greys')
    # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    # fig, ax = plt.subplots()
    # im = ax.pcolormesh(grid[0], grid[1], z, cmap=cmap, norm=norm)
    # fig.colorbar(im, ax=ax)
    # plt.show()
    print(f["obs"][0].shape)
    create_video(f["obs_next"][:50, 0], "video.avi", 1)
    # for i, img in enumerate(f["obs_next"][5]):
    #     cv2.imwrite(f"render/{i}.png",
    #         cv2.cvtColor(
    #             img.transpose(1, 2, 0),
    #             cv2.COLOR_RGB2BGR,
    #         )
    #     )
    #     cv2.waitKey(0)

assert False


env = FrameStack(VisualMazeEnv(maze_spec=LARGE_OPEN), 1)
env.seed(0)
frames = []
obs = env.reset()
frames.append(obs)
# video = cv2.VideoWriter("video.avi", cv2.VideoWriter_fourcc(*"HFYU"), 2, obs[0].shape[1:])
actions = [6] * 100
controller = waypoint_controller.WaypointController(env.str_maze_spec, d_gain=0)
for i, action in enumerate(actions):
    # print(ss_model.encoder(torch.as_tensor(obs, device="cuda").flatten(0, 1).unsqueeze(0)))
    position = env.sim.data.qpos.copy().ravel()
    velocity = env.sim.data.qvel.copy().ravel()
    # act, done = controller.get_action(position, velocity, env.get_target())
    # action = np.argmax(VisualMazeEnv.ACTIONS @ act)
    obs, reward, done, info = env.step(action)
    img = cv2.cvtColor(obs[-1].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    print(velocity, env.sim.data.qvel.ravel())
    # print(act, VisualMazeEnv.ACTIONS[action])
    cv2.imshow("aasdf", img.repeat(10, axis=0).repeat(10, axis=1))
    cv2.waitKey(0)
    # cv2.putText(img, str(env.sim.data.qvel.ravel()), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 0, 0), 2, cv2.LINE_AA)
    # video.write(img)
    frames.append(obs)

# video.release()
assert False
create_video([f[-1] for f in frames], "video.avi", 4)

from nav2d_representation.nav2d.nav2d import Navigate2D
from nav2d_representation import utils



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
env.seed(1)

path = "experiments/offline/ftarget_convenc_e1.0_nogoal/multi_step_312000.pt"
model = torch.load(path)
for i in tqdm(range(100)):
    obs = env.reset()
    agent_hm, obs_hm = utils.grad_heatmap(obs, model.encoder)
    combined = np.concatenate([((obs + 1) / 2).transpose([1, 2, 0]), agent_hm, obs_hm], axis=1)
    plt.clf()
    plt.imshow(combined)
    plt.savefig(f"render/{i}.png")
assert False

path = "experiments/offline/ftarget_convenc_e1.0_nogoal/"
model = torch.load(path + "multi_step_310000.pt")
obs = env.reset()
plt.imshow(utils.perturb_heatmap(obs, model.encoder)[-1].T)
plt.show()
assert False
encoded = ss_model.encoder(
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