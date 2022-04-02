from argparse import ArgumentParser
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, RLock
import h5py

from nav2d_representation.nav2d.nav2d import Navigate2D


def collect(num, idx, seed, epsilon, num_obstacles):
    env = Navigate2D(num_obstacles)
    env.seed(seed + idx)
    np.random.seed(seed + idx)

    buffer = []

    global_step = 0
    pbar = tqdm(total=num, position=idx)
    while global_step < num:
        obs = env.reset()
        obs[2] = -1  # clear goal channel
        done = False
        recompute = True
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                recompute = True
            else:
                if recompute:
                    optimal_actions = env.find_path()
                action = optimal_actions.pop(0)
                recompute = False
            obs_next, rew, done, info = env.step(action)
            obs_next[2] = -1  # clear goal channel
            buffer.append((obs, action, rew, obs_next, done, info["pos"], info["goal"]))
            obs = obs_next
            global_step += 1
            pbar.update(1)

    return buffer


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--num_obstacles", type=int, default=15)
    args = parser.parse_args()
    # with h5py.File(f"nav2d_dataset_seed_{SEED}_eps_{EPSILON}_size_{TOTAL_SIZE}.hdf5", "r") as f:
    #     for i, obs in enumerate(f["obs"]):
    #         print(i)
    #         plt.clf()
    #         plt.imshow(utils.render(obs).T)
    #         plt.savefig(f"render_test/{i:03}.png")

    with Pool(args.num_workers, initargs=(RLock(),), initializer=tqdm.set_lock) as p:
        buffers = p.starmap(
            collect,
            [
                (
                    args.size // args.num_workers,
                    i,
                    args.seed,
                    args.epsilon,
                    args.num_obstacles,
                )
                for i in range(args.num_workers)
            ],
        )
    buffer = [x for b in buffers for x in b]

    with h5py.File(
        f"nav2d_dataset_s{args.seed}_e{args.epsilon}_size{args.size}.hdf5", "w"
    ) as f:
        f["obs"] = np.array([x[0] for x in buffer])
        f["action"] = np.array([x[1] for x in buffer])
        f["reward"] = np.array([x[2] for x in buffer])
        f["obs_next"] = np.array([x[3] for x in buffer])
        f["done"] = np.array([x[4] for x in buffer])
        f["info/pos"] = np.array([x[5] for x in buffer])
        f["info/goal"] = np.array([x[6] for x in buffer])

        f.attrs["num_obstacles"] = args.num_obstacles
        f.attrs["env"] = "nav2d"


if __name__ == "__main__":
    main()
