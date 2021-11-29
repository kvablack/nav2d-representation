from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, RLock
import h5py

import utils
from nav2d import Navigate2D

SEED = 0
TOTAL_SIZE = int(1e6)
EPSILON = 1.0
WORKERS = 16

HYP = {
    "grid_size": 20,
    "obstacle_diameter": 1,
    "scale": 1,
    "min_goal_dist": 10,
    "num_obstacles": 15,
    "use_factorized_state": False,
    "max_episode_length": 50,
}


def collect(num, idx):
    env = Navigate2D(HYP)
    env.seed(SEED + idx)
    np.random.seed(SEED + idx)

    buffer = []

    global_step = 0
    pbar = tqdm(total=num, position=idx)
    while global_step < num:
        obs = env.reset()
        obs[2] = -1
        done = False
        recompute = True
        while not done:
            if np.random.rand() < EPSILON:
                action = env.action_space.sample()
                recompute = True
            else:
                if recompute:
                    optimal_actions = env.find_path()
                action = optimal_actions.pop(0)
                recompute = False
            obs_next, rew, done, info = env.step(action)
            obs_next[2] = -1
            buffer.append((obs, action, rew, obs_next, done, info["pos"], info["goal"]))
            obs = obs_next
            global_step += 1
            pbar.update(1)

    return buffer


def main():
    # with h5py.File(f"nav2d_dataset_seed_{SEED}_eps_{EPSILON}_size_{TOTAL_SIZE}_nogoal.hdf5", "r") as f:
    #     for i, obs in enumerate(f["obs"]):
    #         print(i)
    #         plt.clf()
    #         plt.imshow(utils.render(obs).T)
    #         plt.savefig(f"render_test/{i:03}.png")

    with Pool(WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as p:
        buffers = p.starmap(collect, [
            (TOTAL_SIZE // WORKERS, i) for i in range(WORKERS)
        ])
    buffer = [x for b in buffers for x in b]

    with h5py.File(f"nav2d_dataset_seed_{SEED}_eps_{EPSILON}_size_{TOTAL_SIZE}_nogoal.hdf5", "w") as f:
        f["obs"] = np.array([x[0] for x in buffer])
        f["action"] = np.array([x[1] for x in buffer])
        f["reward"] = np.array([x[2] for x in buffer])
        f["obs_next"] = np.array([x[3] for x in buffer])
        f["done"] = np.array([x[4] for x in buffer])
        f["info/pos"] = np.array([x[5] for x in buffer])
        f["info/goal"] = np.array([x[6] for x in buffer])

        for key, value in HYP.items():
            f.attrs[key] = value


if __name__ == "__main__":
    main()
