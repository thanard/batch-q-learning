import argparse
import os.path
import pickle as pkl
import numpy as np
import torch
import torch.optim as optim
import csv
import pickle
from torchvision.utils import save_image
from vae_model import VAE
from block_env import BlockEnv
from utils import get_embedding, ReplayMemory, DQN, optimize_model, eval_task, from_tensor_to_var, from_numpy_to_pil
from torch.autograd import Variable
from torchvision import datasets, transforms

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
                    default="q-learning-results-image-redo")
parser.add_argument('-state', action="store_true",
                    help="either image or state")
parser.add_argument('-embdist', action="store_true",
                    help="either truedist or embdist")
parser.add_argument('-infdata', action="store_true",
                    help="either smalldata or infdata")
parser.add_argument('-shapedreward', action="store_true",
                    help="either binary or shaped reward")
# parser.add_argument('--n_trajs', type=int, default=301,
#                     help="First n_trajs trajectories are used for actions.")
args = parser.parse_args()

embedding_params = os.path.join(args.path_prefix, args.embedding_params)
test_data = os.path.join(args.path_prefix, args.test_data)
transition_file = os.path.join(args.path_prefix, args.transition_file)
save_path = os.path.join(args.path_prefix, args.save_path)
is_image = not args.state
is_truedist = not args.embdist
is_smalldata = not args.infdata
is_binaryreward = not args.shapedreward
print(["image" if is_image else "state"] +
      ["truedist" if is_truedist else "embdist"] +
      ["2K" if is_smalldata else "38K"] +
      ["binaryreward" if is_binaryreward else ""])
kwargs = {}
if not is_truedist:
    assert is_image
# n_trajs = args.n_trajs

with open(test_data, 'rb') as f:
    test_tasks = pkl.load(f)
raw_transitions = np.load(transition_file)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if is_image:
    model = VAE(image_channels=3, z_dim=10).cuda()
    model.load_state_dict(torch.load(embedding_params))
    kwargs['model'] = model

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

"""
Set up the environment
"""
env = BlockEnv()
env.reset()
env.viewer_setup()

"""
Make transitions
"""
filename = "processed_transition_2k.pkl" if is_smalldata else "processed_transition_38k.pkl"
if os.path.exists(filename):
    with open(filename, 'rb') as fp:
        transitions = pickle.load(fp)
else:
    n_trajs = 330 if is_smalldata else len(raw_transitions)
    print("Number of transitions: %d" % sum([len(raw_transitions[i]) for i in range(n_trajs)]))
    transitions = []
    for i in range(n_trajs):
        for t in range(len(raw_transitions[i]) - 1):
            o, o_next, true_s, true_s_next = raw_transitions[i][t][0], \
                                             raw_transitions[i][t + 1][0], \
                                             raw_transitions[i][t][1]['state'], \
                                             raw_transitions[i][t + 1][1]['state']
            if o.sum() != 0 and o_next.sum() != 0:
                with torch.no_grad():
                    # if not is_image:
                    #     s = [np.array([1])]
                    #     s_next = [np.array([1])]
                    # else:
                    s = model.encode(Variable(torch.cuda.FloatTensor(np.transpose(o, (2, 0, 1))[None])))[1]
                    s_next = model.encode(Variable(torch.cuda.FloatTensor(np.transpose(o_next, (2, 0, 1))[None])))[1]
                    if i == 0 and t == 0:
                        recon = model.decode(s)
                        import ipdb; ipdb.set_trace()
                        save_image(torch.cat([torch.cuda.FloatTensor(np.transpose(o, (2, 0, 1))[None]).cpu()/255, recon.data.cpu()], dim=0),
                                   os.path.join(save_path, 'verified_img_preprocessing.png'))
                    # import ipdb; ipdb.set_trace()
                    # s = model.encode(from_tensor_to_var(transform(from_numpy_to_pil(o)))[None, :])[1]
                    # s_next = model.encode(from_tensor_to_var(transform(from_numpy_to_pil(o_next)))[None, :])[1]
                    # if i == 0 and t == 0:
                    #     recon = model.decode(s)
                    #     save_image(torch.cat([from_tensor_to_var(transform(from_numpy_to_pil(o)))[None, :], recon], dim=0).data.cpu(),
                    #                os.path.join(save_path, 'verified_img_preprocessing.png'))

                transitions.append((o, o_next,
                                    s.cpu().numpy()[0].astype(np.float64),
                                    s_next.cpu().numpy()[0].astype(np.float64),
                                    true_s[1:, :2].reshape(-1),
                                    true_s_next[1:, :2].reshape(-1)))
    # Save transitions.
    with open(filename, 'wb') as fp:
        pickle.dump(transitions, fp)

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
TARGET_UPDATE = 20
d_state = 10
d_action = 4
is_embdist = not is_truedist
is_shapedreward = not is_binaryreward
first_task, last_task = 40, 49

for i_task, (start, goal) in enumerate(test_tasks):
    if i_task < first_task or i_task > last_task:
        continue
    print("\n\n### Task %d ###" % i_task)
    """
    Set up
    """
    if is_embdist:
        env.reset(goal)
        o_goal = env.render(mode='rgb_array')
        s_goal = get_embedding(o_goal, model).cpu().numpy()
        kwargs["emb_goal"] = s_goal

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
        if is_embdist:
            rad = np.linalg.norm(s_next - kwargs["emb_goal"], 2)
            threshold = 3.5
            kwargs["emb_threshold"] = threshold
        else:
            rad = np.linalg.norm(ts_next - goal.reshape(-1), 2)
            threshold = 0.5
        r = -1
        if rad < threshold:
            count += 1
            # print(ts_next)
            r = 0
            s_next = None
        if is_shapedreward:
            r -= rad
        import ipdb; ipdb.set_trace()
        memory.push(s, a, s_next, r)
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
            if it % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        pred_v, real_dist, emb_dist, reward, emb_reward = eval_task(env,
                                                          policy_net, start,
                                                          goal, i_task,
                                                          **kwargs)
        print("Epoch %d:: avg loss: %.3f, pred v: %.3f, real dist: %.3f, reward: %d" %
              (epoch, loss / n_iters, pred_v, real_dist, reward))
        with open(os.path.join(save_path, "%d/log.csv" % i_task), 'a') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(["epoch",
                                 "avg loss",
                                 "predicted v",
                                 "real distance",
                                 "embed distance",
                                 "reward",
                                 "embed reward"])
            writer.writerow([epoch, loss / n_iters,
                             pred_v, real_dist, emb_dist,
                             reward, emb_reward])

    """
    Saving results
    """
    pred_v, real_dist, emb_dist, reward, emb_reward = eval_task(env, policy_net, start,
                                                      goal, i_task,
                                                      save_path=save_path,
                                                      is_render=True,
                                                      **kwargs)
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
                             "final embed distance",
                             "final reward",
                             "final embed reward"])
        writer.writerow([i_task, pred_v, real_dist,
                         emb_dist, reward, emb_reward])
