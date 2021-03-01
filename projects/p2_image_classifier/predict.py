import argparse
import utils
import modelfuncs

parser = argparse.ArgumentParser()

parser.add_argument(
    "--image_path", type=str, help="image path")
parser.add_argument(
    "--checkpoint", type=str, help="checkpoint file name")
parser.add_argument(
    "--topk", type=str, default= 3, help="return top k most likely classes")
parser.add_argument(
    "--GPU", help="use GPU for predicting", action="store_true")

args = parser.parse_args()

if args.GPU:
    print('GPU in use')

device = 'cuda' if args.GPU else 'cpu'

if args.image_path is None:
    args.image_path = utils.sample_image()

model, optimizer = modelfuncs.load_checkpoint(args.checkpoint, device)
modelfuncs.display_preds(args.image_path, model, args.topk, device)
