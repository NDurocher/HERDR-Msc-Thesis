from torch import nn
import torch
import cv2
import sys
from pathlib import Path
from torchvision import transforms
from openvino.inference_engine import IECore, IENetwork, Blob, TensorDesc

dir_name = Path(Path.cwd()).parent.parent.parent
sys.path.insert(1, str(dir_name)+'/src')
from Badgrnet import HERDR
from actionplanner import HERDRPlan

if __name__ == "__main__":
    def impreprosses(im):
        im = cv2.resize(im, (320, 240))
        im = loader(im).float()
        im = im.unsqueeze(0)
        im = im.repeat(batches, 1, 1, 1)
        return im


    batches = 50
    hor = 20
    planner = HERDRPlan(Horizon=hor)
    # model = HERDR(Horizon=hor, RnnDim=64)
    # model = torch.load("/Users/NathanDurocher/Documents/GitHub/HERDR/Test/controllers/Hircus/Herdr_cross.pth",
    #                    map_location=torch.device('cpu'))
    # net = IENetwork(model='Herdr.xml', weights='Herdr.bin')
    ie = IECore()
    net = IECore.read_network(ie, 'Herdr.xml', 'Herdr.bin')
    output_blob = next(iter(net.outputs))
    exec_net = ie.load_network(network=net, device_name='MYRIAD', num_requests=1)
    inference_request = exec_net.requests[0]

    # print(output_blob)
    video = cv2.VideoCapture(0)

    # Clear camera opening frame
    _, _ = video.read()

    # Take one image
    loader = transforms.Compose([transforms.ToTensor()])

    for i in range(0, 1):
    # while True:
        check, frame = video.read()
        frame = impreprosses(frame)
        actions = planner.sample_new(batches=batches)
        # print(frame.shape, actions.shape)
        # frame_desc = TensorDesc(precision='FP32', dims=[1, 3, 240, 320], layout='NCHW')
        # actions_desc = TensorDesc(precision='FP32', dims=[1, 20, 2], layout='CHW')
        # input_blob = Blob([frame_desc, actions_desc], [frame, actions])
        # input_blob_name = next(iter(input_blob))
        ### ONLY WORKS WITH BATCHES OF 50 - would have to re convert torch to onnx to openvino for more batches
        output = exec_net.infer(inputs={"input.1": frame, "input.25": actions})
        # inference_request.set_blob(blob_name=input_blob_name, blob=input_blob)
        # res = inference_request.infer()
        # output_blob_name = next(iter(output_blob))
        # output = inference_request.output_blob[output_blob_name].buffer
        print(type(output[output_blob]))

        # print(r.flatten(start_dim=0, end_dim=1).shape)
        # torch.onnx.export(model,(frame, actions),'Herdr.onnx')
        # print("Prediction: ", r.detach().flatten())
        # print(res)
        # planner.update_new(r.detach(), actions)
