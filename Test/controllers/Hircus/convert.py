import torch
from torch import nn
from pathlib import Path
import sys
dir_name = Path(Path.cwd()).parent.parent.parent
sys.path.insert(1, str(dir_name)+'/src')
from Badgrnet import HERDR

if __name__ == '__main__':
    model = torch.load(
        "Herdr_Feb22_640_2Hr.pth",
        map_location=torch.device('cpu'))
    model.model_out = nn.Sequential(
        model.model_out,
        nn.Sigmoid()
    )
    model.eval()
    frame = torch.zeros((1, 3, 480, 640))
    actions = torch.zeros((1, 10, 2))

    torch.onnx.export(model, (frame, actions), 'Herdr.onnx', input_names=['img', 'actions'], verbose=True)