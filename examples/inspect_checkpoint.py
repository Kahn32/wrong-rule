import argparse
import torch

from wrongrule.models import MLP
from wrongrule.training import TrainConfig

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
args = parser.parse_args()
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
config = TrainConfig(**payload["train_config"])
model = MLP(payload["dim"], config.width, config.depth, config.activation, config.init_scale)
model.load_state_dict(payload["state_dict"])
model.eval()
x = torch.zeros(2, payload["dim"])
x[1, 0] = .5
with torch.no_grad():
    print("Inputs:", x)
    print("Predictions:", model(x))
