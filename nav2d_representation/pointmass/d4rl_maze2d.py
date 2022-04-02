from collections import deque
from d4rl.pointmaze import MazeEnv, U_MAZE, LARGE_MAZE, MEDIUM_MAZE
import numpy as np
from gym import spaces
from gym.wrappers import LazyFrames


class VisualMazeEnv(MazeEnv):
    ACTIONS = np.array(
        [[-1, 0], [0, -1], [0, 1], [1, 0], [1, 1], [-1, -1], [-1, 1], [1, -1]]
    )
    RENDER_SIZE = 80

    def __init__(
        self,
        maze_spec=LARGE_MAZE,
        reward_type="sparse",
        reset_target=True,
        max_timesteps=120,
        frame_skip=5,
        **kwargs
    ):
        self.step_count = 0
        self.max_timesteps = max_timesteps
        self.initialized = False
        super().__init__(maze_spec, reward_type, reset_target, **kwargs)
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.RENDER_SIZE, self.RENDER_SIZE),
            dtype=np.uint8,
        )
        self.frame_skip = frame_skip
        self.position_buffer = []

        self.sim.model.vis.global_.fovy = 5
        self.sim.model.vis.map.zfar = 2000

    def reset_model(self):
        self.step_count = 0
        self.initialized = True
        self.position_buffer = []
        super().reset_model()
        # self.set_marker()  # required to update goal pos visually
        return self._get_obs()

    def step(self, action):
        if not self.initialized:
            return self._get_obs(), 0, False, {}

        self.position_buffer.append(self.sim.data.qpos.ravel().copy())
        self.step_count += 1
        self.clip_velocity()
        self.do_simulation(self.ACTIONS[action], self.frame_skip)
        # self.set_marker()
        ob = self._get_obs()
        if self.reward_type == "sparse":
            reward = (
                0.0
                if np.linalg.norm(self.sim.data.qpos.ravel() - self._target) <= 0.5
                else -1.0
            )
        elif self.reward_type == "dense":
            reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
        else:
            raise ValueError("Unknown reward type %s" % self.reward_type)

        done = reward == 0 or self.step_count >= self.max_timesteps

        return ob, reward, done, {}

    def _get_obs(self):
        return self.render(
            mode="rgb_array", width=self.RENDER_SIZE, height=self.RENDER_SIZE
        ).transpose(2, 0, 1)

    def viewer_setup(self):
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0
        self.viewer.cam.distance = 125
        self.viewer.cam.lookat[:] = [5, 6.5, 0]

    def set_marker(self):
        # the coordinates are offset by 1.2 for some reason...
        self.data.site_xpos[self.model.site_name2id("target_site")] = np.array(
            [self._target[0] + 1.2, self._target[1] + 1.2, 0.0]
        )

    def her(self, frame_stack=1):
        fake_target = self.position_buffer[-1]
        self.set_target(fake_target)

        self.set_state(self.position_buffer[0], np.zeros(2))
        self.set_marker()
        first_obs = self._get_obs()
        frames = deque([first_obs for _ in range(frame_stack)], maxlen=frame_stack)

        ret = [LazyFrames(list(frames))]
        for pos in self.position_buffer[1:]:
            self.set_state(pos, np.zeros(2))
            self.set_marker()
            frames.append(self._get_obs())
            ret.append(LazyFrames(list(frames)))

        return ret
