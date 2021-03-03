import argparse
import utils
import modelfuncs

parser = argparse.ArgumentParser()

parser.add_argument(
    "--image_path", type=str,
    help="Image path. If none, random image chosen from test folder.")
parser.add_argument(
    "--checkpoint", type=str,
    help="name of checkpoint file in model_checkpoints\\arch\\ dir")
parser.add_argument(
    "--topk", type=int, default= 3,
    help="return top k most likely classes")
parser.add_argument(
    "--GPU",  action="store_true", help="use GPU for predicting")

args = parser.parse_args()

if args.GPU:
    print('GPU in use')

device = 'cuda' if args.GPU else 'cpu'

if args.image_path is None:
    args.image_path = utils.sample_image()

model, optimizer = modelfuncs.load_checkpoint(args.checkpoint, device)
modelfuncs.display_preds(args.image_path, model, args.topk, device)
