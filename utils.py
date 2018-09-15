import matplotlib.pyplot as plt
import scipy.misc
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import numpy as np
from collections import namedtuple
from torch.autograd import Variable
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def render_image(env, path):
    img = env.render(mode='rgb_array')
    img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
    plt.imsave(path, img)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, d_state=4, d_action=4):
        super(DQN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(d_state + d_action, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.main(x)


def max_actions(target_net, states, n_actions=200, d_action=4, max_size=0.1):
    rand_sam = Variable(torch.rand(n_actions, 2)*max_size*2 - max_size).cuda()
    # actions = Variable(torch.zeros(2*n_actions, 6)).cuda()
    # actions[:n_actions, :2] = rand_sam
    # actions[n_actions:, 3:5] = rand_sam
    actions = Variable(torch.zeros(2*n_actions, d_action)).cuda()
    actions[:n_actions, :2] = rand_sam
    actions[n_actions:, 2:] = rand_sam
    max_value_from_states = []
    max_action_from_states = []
    for s in states:
        batch = torch.cat([s[None].repeat(2*n_actions, 1), actions], dim=1)
        max_value_from_s, max_action_from_s = target_net(batch).max(0)
        max_value_from_states.append(max_value_from_s.detach().data[0])
        max_action_from_states.append(actions[max_action_from_s.detach().data[0]])
    return Variable(torch.cuda.FloatTensor(max_value_from_states)), \
           torch.cat(max_action_from_states)


def optimize_model(memory,
                   policy_net,
                   target_net,
                   optimizer,
                   GAMMA,
                   BATCH_SIZE):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = [i for i, s in enumerate(batch.next_state) if s is not None]
    non_final_next_states = Variable(torch.FloatTensor([s for s in batch.next_state
                                                if s is not None]).cuda())
    state_batch = Variable(torch.cuda.FloatTensor(batch.state), requires_grad=False)
    action_batch = Variable(torch.cuda.FloatTensor(batch.action), requires_grad=False)
    reward_batch = Variable(torch.cuda.FloatTensor(batch.reward), requires_grad=False)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(torch.cat([state_batch, action_batch], dim=1))

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE), requires_grad=False).cuda()
    if len(non_final_mask) > 0:
        next_state_values[non_final_mask] = max_actions(target_net, non_final_next_states)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # loss = ((state_action_values - expected_state_action_values.unsqueeze(1))**2).sum()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.data[0]


def get_embedding(img, model):
    """
    :param img: raw image from Mujoco render
    :param model: an embedding model
    :return: an embedding vector in torch Variable
    """
    o_goal = scipy.misc.imresize(img,
                                 (64, 64, 3),
                                 interp='nearest')
    with torch.no_grad():
        return model.encode(Variable(torch.cuda.FloatTensor(np.transpose(o_goal, (2, 0, 1))[None])))[1]


def eval_task(env, policy_net, start, goal, i_task,
              save_path=None, is_render=False, model=None,
              emb_goal=None, emb_threshold=None):
    env.reset(start)
    np_cur_s = start
    cur_s = None
    reward = 0
    emb_reward = 0
    for i in range(30):
        if is_render:
            env.render()
        if save_path is not None:
            if not os.path.exists(os.path.join(save_path, "%d" % i_task)):
                os.makedirs(os.path.join(save_path, "%d" % i_task))
            render_image(env, os.path.join(save_path, "%d/%d.png" % (i_task, i)))
        if model is not None:
            if cur_s is None:
                cur_s = get_embedding(env.render(mode='rgb_array'), model)
            # There is some stochasticity in rendering images.
            # assert torch.eq(cur_s,
            #                 get_embedding(env.render(mode='rgb_array'), model))
        else:
            cur_s = Variable(torch.zeros(1, 4)).cuda()
            cur_s[0, :2] = Variable(torch.cuda.FloatTensor(np_cur_s[0]))
            cur_s[0, 2:] = Variable(torch.cuda.FloatTensor(np_cur_s[1]))
        cur_v, cur_a = max_actions(policy_net, cur_s)
        np_cur_s = env.step_only(cur_a.cpu().numpy())[1:, :2]
        rad = np.linalg.norm(np_cur_s - goal, 2)
        if rad > 0.5:
            reward -= 1
        if emb_goal is not None:
            assert emb_threshold is not None
            cur_s = get_embedding(env.render(mode='rgb_array'), model)
            np_emb_cur_s = cur_s.cpu().numpy()
            emb_rad = np.linalg.norm(np_emb_cur_s - emb_goal, 2)
            if emb_rad > emb_threshold:
                emb_reward -= 1

    pred_v = cur_v.data[0]
    real_dist = np.linalg.norm(np_cur_s - goal, 2)
    emb_dist = 0 if emb_goal is None else emb_rad
    if save_path:
        env.reset(goal)
        env.render()
        render_image(env, os.path.join(save_path, "%d/goal.png" % i_task))
    return pred_v, real_dist, emb_dist, reward, emb_reward
