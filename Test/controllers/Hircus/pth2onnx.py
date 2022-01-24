import onnx
import torch
import sys
from pathlib import Path

dir_name = Path(Path.cwd()).parent.parent.parent
sys.path.insert(1, str(dir_name)+'/src')
from Badgrnet import HERDR

if __name__ == "__main__":
    model_name = "Herdr_cross04-01-2022--12 17 14.pth"
    model = torch.load(model_name, map_location=torch.device('cpu'))
    frame = 0.1*torch.ones(1, 3, 240, 320)
    actions = 0.1*torch.ones(1, 20, 2)
    torch.onnx.export(model, (frame, actions), 'Herdr.onnx', export_params=True, opset_version=10)
    onnx_model = onnx.load('Herdr.onnx')
    graph = onnx_model.graph
    print(onnx.checker.check_model(onnx_model))
    # print(onnx.helper.printable_graph(graph))
    # ls = [0,1,3,4,6,7,8]
    # for item in range(0, 1):
    #     graph.node.remove(graph.node[item])
        # print(item)
    # onnx.save(onnx_model, 'Herdr_pruned.onnx')