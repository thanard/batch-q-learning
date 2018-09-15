import numpy as np
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.mujoco_py import MjViewer

num_blocks = 2
block_size = .12
loc = (1.5 - block_size * 2)


class BlockEnv(MujocoEnv, Serializable):
    global FILE

    def __init__(self, *args, **kwargs):
        FILE = './block_stack_{}.xml'.format(num_blocks)
        # FILE = './block_stack_2.xml'
        super(BlockEnv, self).__init__(*args, **kwargs, file_path=FILE)
        Serializable.quick_init(self, locals())

    def reset(self, init_state=None):
        """
        :param init_state: (2, 2) positions for the first and the second block.
        :return: set geom_xpos
        """
        if init_state is None:
            init_state = (np.random.rand(num_blocks, 2) * loc) - (loc / 2)
        init_pos = np.zeros((num_blocks * 3))
        init_pos[:2] = init_state[0] - self.model.body_pos[1, :2]
        init_pos[3:5] = init_state[1] - self.model.body_pos[2, :2]
        full_state = np.concatenate([init_pos, self.init_qvel.squeeze(),
                      self.init_qacc.squeeze(), self.init_ctrl.squeeze()])
        obs = super(BlockEnv, self).reset(init_state=full_state)
        return obs

    def get_current_obs(self):
        return self.model.data.geom_xpos

    def viewer_setup(self, config=None):
        viewer = self.get_viewer(config=config)
        viewer.cam.trackbodyid = 0
        viewer.cam.distance = 2.75
        viewer.cam.elevation = -60

    def step(self, action):
        done = False
        reward = 0
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        return Step(next_obs, reward, done)

    def step_only(self, action):
        next_state = self.get_current_obs()[1:, :2] + action.reshape(2, 2)
        return self.reset(init_state=next_state)

    def get_viewer(self, config):
        if self.viewer is None:
            self.viewer = MjViewer(visible=False)
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer
