import argparse
import os.path
import pickle as pkl
import numpy as np
import torch
import torch.optim as optim
import csv
from vae_model import VAE
from block_env import BlockEnv
from utils import render_image, ReplayMemory, DQN, optimize_model, eval_task
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--path_prefix", type=str,
                    default="/home/thanard/Downloads/block-action-data/")
parser.add_argument('--embedding_params', type=str,
                    default="z-dim-10/vae_2000.pth")
parser.add_argument('--test_data', type=str,
                    default="quant_data_paper.pkl")
parser.add_argument('--transition_file', type=str,
                    default="randact_s0.12_2_data_10000.npy")
parser.add_argument('--save_path', type=str,
                    default="q-learning-results")
# parser.add_argument('--n_trajs', type=int, default=301,
#                     help="First n_trajs trajectories are used for actions.")
args = parser.parse_args()

embedding_params = os.path.join(args.path_prefix, args.embedding_params)
test_data = os.path.join(args.path_prefix, args.test_data)
transition_file = os.path.join(args.path_prefix, args.transition_file)
save_path = os.path.join(args.path_prefix, args.save_path)

# n_trajs = args.n_trajs

with open(test_data, 'rb') as f:
    test_tasks = pkl.load(f)
raw_transitions = np.load(transition_file)
if not os.path.exists(save_path):
    os.makedirs(save_path)

model = VAE(image_channels=3, z_dim=10).cuda()
model.load_state_dict(torch.load(embedding_params))

"""
Set up the environment
"""
env = BlockEnv()
env.reset()
env.render()
env.viewer_setup()
render_image(env, os.path.join(save_path, "1.png"))
env.reset(raw_transitions[0][0][1]['state'][1:, :2])

"""
Make transitions
"""
n_trajs = 330
# n_trajs = len(raw_transitions)
print("Number of transitions: %d" % sum([len(raw_transitions[i]) for i in range(n_trajs)]))
transitions = []
for i in range(n_trajs):
    for t in range(len(raw_transitions[i])-1):
        o, o_next, true_s, true_s_next = raw_transitions[i][t][0],\
                                         raw_transitions[i][t+1][0], \
                                         raw_transitions[i][t][1]['state'],\
                                         raw_transitions[i][t+1][1]['state']
        if o.sum() != 0 and o_next.sum() != 0:
            with torch.no_grad():
                s = [np.array([1])]
                s_next = [np.array([1])]
                # s = model.encode(Variable(torch.cuda.FloatTensor(np.transpose(o, (2, 0, 1))[None])))[0].cpu().numpy()
                # s_next = model.encode(Variable(torch.cuda.FloatTensor(np.transpose(o_next,(2,0,1))[None])))[0].cpu().numpy()
            transitions.append((o, o_next, s[0], s_next[0], true_s[1:, :2].reshape(-1), true_s_next[1:, :2].reshape(-1)))
print("o shape: ", transitions[0][0].shape)
print("s embedding shape: ", transitions[0][2].shape)
print("true s shape: ", transitions[0][4].shape)
print("Number of training transitions: %d" % len(transitions))

"""
Batch Q-Learning
"""
N_EPOCHS = 100
BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 1
d_state = 4
d_action = 4

for i_task, (start, goal) in enumerate(test_tasks):
    print("\n\n### Task %d ###" % i_task)

    """
    Set up
    """
    policy_net = DQN(d_state, d_action).cuda()
    target_net = DQN(d_state, d_action).cuda()

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(len(transitions))

    if not os.path.exists(os.path.join(save_path, "%d" % i_task)):
        os.makedirs(os.path.join(save_path, "%d" % i_task))

    """
    Push data into memory
    """
    count = 0
    for trans in transitions:
        o, o_next, s, s_next, ts, ts_next = trans
        a = ts_next - ts
        rad = np.linalg.norm(ts_next - goal.reshape(-1), 2)
        r = -1
        if rad < 0.5:
            count += 1
            # print(ts_next)
            r = 0
            ts_next = None
        memory.push(ts, a, ts_next, r)
    print("Number of goals reached in transitions: %d" % count)

    """
    Training Q-function
    """
    n_iters = len(transitions) // BATCH_SIZE
    for epoch in range(N_EPOCHS):
        loss = 0
        for it in range(n_iters):
            loss += optimize_model(memory,
                                   policy_net,
                                   target_net,
                                   optimizer,
                                   GAMMA,
                                   BATCH_SIZE)
        pred_v, real_dist, reward = eval_task(env, policy_net, start, goal, i_task)
        print("Epoch %d:: avg loss: %.3f, pred v: %.3f, real dist: %.3f, reward: %d" %
              (epoch, loss / n_iters, pred_v, real_dist, reward))
        with open(os.path.join(save_path, "%d/log.csv" % i_task), 'a') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(["epoch",
                                 "avg loss"
                                 "predicted v",
                                 "real distance",
                                 "reward"])
            writer.writerow([epoch, loss / n_iters,
                             pred_v, real_dist, reward])
        if epoch % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    """
    Saving results
    """
    pred_v, real_dist, reward = eval_task(env, policy_net, start, goal, i_task,
                                  save_path=save_path, is_render=True)
    print("final predicted v: ", pred_v)
    print("final real distance: ", real_dist)
    print("final reward: ", reward)
    torch.save(policy_net.state_dict(), os.path.join(save_path, '%d/policy_net.pt' % i_task))
    with open(os.path.join(save_path, "meta-log.csv"), 'a') as f:
        writer = csv.writer(f)
        if i_task == 0:
            writer.writerow(["task",
                             "final predicted v",
                             "final real distance",
                             "final reward"])
        writer.writerow([i_task, pred_v, real_dist, reward])
