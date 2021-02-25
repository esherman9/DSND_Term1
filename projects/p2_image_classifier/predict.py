import argparse

parser = argparse.ArgumentParser()
parser.add_argument("/path/to/image", type=str, help="image path")
parser.add_argument("checkpoint", type=str, help="path of model checkpoint")
parser.add_argument("--topk", type=str, help="return top k most likely classes")
parser.add_argument("--GPU", help="use GPU for training", action="store_true")

args = parser.parse_args()
if args.GPU:
    print("GPU in use")

preds = predict(args.'/path/to/image')
