import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
import argparse
import doodad
import os
from doodad.mode import EC2AutoconfigDocker
from doodad.utils import REPO_DIR

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
                    default="q-learning-results-image")
parser.add_argument('-state', action="store_true",
                    help="either image or state")
parser.add_argument('-embdist', action="store_true",
                    help="either truedist or embdist")
parser.add_argument('-infdata', action="store_true",
                    help="either smalldata or infdata")
parser.add_argument('-shapedreward', action="store_true",
                    help="either binary or shaped reward")
parser.add_argument('-ec2', action="store_true")
args = parser.parse_args()

embedding_params = os.path.join(args.path_prefix, args.embedding_params)
test_data = os.path.join(args.path_prefix, args.test_data)
transition_file = os.path.join(args.path_prefix, args.transition_file)
save_path = os.path.join(args.path_prefix, args.save_path)
is_image = not args.state
is_truedist = not args.embdist
is_smalldata = not args.infdata
is_binaryreward = not args.shapedreward
exp_name = "batch-q-learning/" + "-".join(
    ["image" if is_image else "state"] +
    ["truedist" if is_truedist else "embdist"] +
    ["binaryreward" if is_binaryreward else ""]
)

if args.ec2:
    run_mode = EC2AutoconfigDocker(
        image='thanard/matplotlib:latest',
        region='us-east-1',
        instance_type='p2.xlarge',
        spot_price=1.0,
        s3_log_prefix=exp_name,
        gpu=True,
        terminate=True,
    )