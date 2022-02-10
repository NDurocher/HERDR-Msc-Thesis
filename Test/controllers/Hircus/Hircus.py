import pandas as pd
from controller import Supervisor
import torch
from torch import nn
from torchvision.utils import save_image
# from torchvision.transforms.functional import resize
import sys
import os
import numpy as np
import h5py
# from transforms3d.euler import mat2euler, axangle2euler
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from random import uniform
from dataclasses import make_dataclass
import optparse
import cv2


dir_name = Path(Path.cwd()).parent.parent.parent
sys.path.insert(1, str(dir_name)+'/src')
from Badgrnet import HERDR
from actionplanner import HERDRPlan
from metrics_utils import plot_trajectory, plot_actions, plot_action_cam_view


def new_goal():
    global GOAL
    GOAL = np.broadcast_to([uniform(-6, 6), uniform(-6, 6)], (BATCH, 2))
    print(GOAL[0])
    pass


def add2pickle(file_name, dataframe, overwrite=False):
    if os.path.isfile(f'./{file_name}') and not overwrite:
        pkl_dataframe = pd.read_pickle(file_name)
        pkl_dataframe = pkl_dataframe.append(dataframe, ignore_index=True)
        pkl_dataframe.to_pickle(file_name, protocol=4)
    else:
        dataframe.to_pickle(file_name, protocol=4)
    pass


class Hircus (Supervisor):
    """Control a Hircus PROTO."""

    def __init__(self, train=True, accel=False, ultra=False):
        """Constructor: initialize constants."""
        Supervisor.__init__(self)
        self.df = pd.DataFrame(columns=["Actions", "Ground_Truth", "Image_Name"])
        self.stateR_df = pd.DataFrame(columns=["State", "Event_Prob", "Target_Pos"])
        self.train = train
        self.accel = accel
        self.is_ultra = ultra
        self.peds = []
        self.state = []
        i = 0
        while not self.getFromDef("Ped%d" % i) is None:
            self.peds.append(self.getFromDef("Ped%d" % i))
            i += 1
        self.logger = self.getFromDef("Logger")
        self.logger.getField('translation').setSFVec3f([GOAL[0, 0, 0], 10, GOAL[0, 0, 1]])
        self.customdata = self.logger.getField('customData')
        self.hircus = self.getSelf()
        self.pose = self.hircus.getPose()
        self.rot = None
        self.frame = None
        self.obj = []
        self.recog = []
        self.actions = torch.tensor([])
        self.now = []
        self.event = torch.tensor([])

        self.load_webots_devices()
        self.enable_sensors(DEVICE_SAMPLE_TIME)
        self.reset_motor_position()
        self.reset_motor_speed()

        self.stamp = 0.
        self.front_speed = 0.
        self.rear_speed = 0.
        self.left_speed = 0.
        self.right_speed = 0.
        self.front_steering_angle = 0.
        self.rear_steering_angle = 0.
        self.wheelbase = 0.38

        self.key = 0

        self.planner = HERDRPlan(Horizon=HRZ, vel_init=0.7)
        self.infer = self.set_infer()

    def set_infer(self):
        if self.train:
            if self.is_ultra:
                self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.yolo.classes = [0]
                return self.yolo_recognize
            else:
                return self.recognize
        else:
            if self.accel:
                return self.accel_infer
            else:
                self.net = torch.load('Herdr_cross06-01-2022--18 50 17.pth', map_location=torch.device('cpu'))
                self.net.model_out = nn.Sequential(
                    self.net.model_out,
                    nn.Sigmoid()
                )
                self.net.eval()
                return self.model_infer

    def accel_infer(self):
        output = exec_net.infer(inputs={"img": self.frame, "actions": self.actions})
        self.event = torch.tensor(output[output_blob]).squeeze(2)

    def model_infer(self):
        self.event = self.net(self.frame, self.actions)[:, :, 0].detach()

    def load_webots_devices(self):
        self.left_motor = self.getDevice('left_wheel')
        self.right_motor = self.getDevice('right_wheel')
        self.front_motor = self.getDevice('front_wheel')
        self.rear_motor = self.getDevice('rear_wheel')
        self.front_steer = self.getDevice('front_steer')
        self.rear_steer = self.getDevice('rear_steer')
        self.gps = self.getDevice('gps')
        self.body_imu = self.getDevice('body_gyro')
        self.gnss_heading_device = self.getDevice('gnss_heading')
        self.camera = self.getDevice('CAM')
        self.Keyboard = self.getKeyboard()
        self.ultra = self.getDevice('distance sensor')

    def enable_sensors(self, step_time):
        self.gps.enable(step_time)
        self.front_steer.getPositionSensor().enable(step_time)
        self.rear_steer.getPositionSensor().enable(step_time)
        self.body_imu.enable(step_time)
        self.gnss_heading_device.enable(step_time)
        self.Keyboard.enable(step_time)
        if not isinstance(self.ultra, type(None)):
            self.ultra.enable(step_time)
        if not isinstance(self.camera, type(None)):
            self.camera.enable(step_time)
            if not self.is_ultra:
                self.camera.recognitionEnable(step_time)
            self.height = self.camera.getHeight()
            self.width = self.camera.getWidth()
            self.fl = self.camera.getFocalLength()

    def reset_motor_position(self):
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.front_motor.setPosition(float('inf'))
        self.rear_motor.setPosition(float('inf'))
        self.front_steer.setPosition(float('inf'))
        self.rear_steer.setPosition(float('inf'))

    def reset_motor_speed(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.front_motor.setVelocity(0)
        self.rear_motor.setVelocity(0)
        self.front_steer.setVelocity(0)
        self.rear_steer.setVelocity(0)

    def update_motors(self, speed, steer):
        if np.isnan(speed) or np.isnan(steer):
            self.reset()
        self.left_motor.setVelocity(speed/WHEEL_RADIUS)
        self.right_motor.setVelocity(speed/WHEEL_RADIUS)
        self.front_motor.setVelocity(speed/WHEEL_RADIUS)
        self.rear_motor.setVelocity(speed/WHEEL_RADIUS)

        self.front_steer.setPosition(-steer)
        self.rear_steering_angle = steer
        self.rear_steer.setPosition(steer)
        self.front_steer.setVelocity(2)
        self.rear_steer.setVelocity(2)

    def yolo_recognize(self):
        self.event = torch.zeros((BATCH, HRZ))
        frame = self.frame[0].permute([1, 2, 0])  #.unsqueeze(0)
        yolo_out = self.yolo(frame.numpy())
        yolo_out = yolo_out.xyxy[0]
        # print(yolo_out)
        # if yolo_out is not None:
        try:
            x = torch.mean(torch.tensor([yolo_out[0, 0], yolo_out[0, 2]]))
            y = torch.mean(torch.tensor([yolo_out[0, 1], yolo_out[0, 3]]))
            image_bytes = self.ultra.getRangeImage(data_type="buffer")
            image_np = np.frombuffer(image_bytes, dtype=np.float32)
            image_np = np.reshape(image_np, (self.height, self.width, 1), order='C')
            dist = image_np[int(y.item()), int(x.item())]
            X = ((1280 / 2 - x) / self.fl) * dist
            obj_pos = torch.tensor([X, dist]).repeat([BATCH, HRZ, 1])
            hircus_pos = self.yolo_calculate_position()
            self.event = self.yolo_is_safe(hircus_pos, obj_pos)
        except:
        # else:
            pass

    def yolo_calculate_position(self):
        pos = torch.zeros(self.actions.shape)
        omega = torch.zeros(self.actions.shape[0])
        for i, val in enumerate(pos.transpose(1, 0)):
            if i + 1 == pos.shape[1]:
                break
            pos[:, i + 1, 0] = pos[:, i, 0] - (WEBOTS_STEP_TIME / SCALE) * torch.sin(omega) * self.actions[:, i, 0]
            pos[:, i + 1, 1] = pos[:, i, 1] + (WEBOTS_STEP_TIME / SCALE) * torch.cos(omega) * self.actions[:, i, 0]
            omega = omega - (WEBOTS_STEP_TIME / SCALE) * self.actions[:, i, 1] * self.actions[:, i, 0] / self.wheelbase
        return pos

    def yolo_is_safe(self, pos, obj_pos):
        boundary = torch.sqrt( torch.square(pos[:, :, 0] - obj_pos[:, :, 1]) + torch.square(pos[:, :, 1] - obj_pos[:, :, 0]) )
        check = boundary < 2
        return check.int()

    def recognize(self):
        self.event = torch.zeros((BATCH, HRZ))
        self.recog = self.camera.getRecognitionObjects()
        # dist = self.ultra.getValue()
        # print(dist)
        if self.recog:
            obj = self.camera.getRecognitionObjects()
            self.obj = [self.getFromId(node.get_id()) for node in obj]
            try:
                for ped in self.obj:
                    ped_pos = torch.tensor(ped.getPosition())
                    ped_pos = ped_pos.repeat(BATCH, HRZ, 1)
                    SFRot = ped.getField("rotation").getSFRotation()
                    ped_ori = torch.tensor(SFRot[0:3]) * SFRot[3]
                    self.event = torch.logical_or(self.is_safe(ped_pos, ped_ori), self.event).float()
            except:
                pass
        else:
            pass

    def reward(self):
        self.rot = self.gnss_heading_device.getRollPitchYaw()
        self.state = self.calculate_position(self.rot)
        #  goalreward Shape: [BATCH, HRZ]
        goalReward = torch.sqrt(torch.square((self.state[:, :, 0]-GOAL[:, :, 0])) +
                                torch.square((self.state[:, :, 2]-GOAL[:, :, 1])))
        self.infer()
        plot_action_cam_view(self.actions, self.frame[0], self.event,
                             self.rear_steering_angle, self.front_motor.getVelocity()*WHEEL_RADIUS)
        event_gain = goalReward.mean()*1.1
        reward = goalReward + event_gain * self.event
        return reward

    def calculate_position(self, rot):
        new_pos = torch.tensor(self.hircus.getPosition())
        new_state = torch.cat((new_pos.T, torch.tensor([-rot[2]])))
        batch_state = new_state.repeat(BATCH, HRZ, 1).transpose(1,2)
        # Y-axis is vertical, movement is in X-Z plane
        # [X Y Z Phi]
        for i in range(0, HRZ - 1):
            batch_state[:, 0, i + 1] = batch_state[:, 0, i] - (WEBOTS_STEP_TIME / SCALE) * torch.cos(
                batch_state[:, 3, i]) * self.actions[:, i, 0]
            batch_state[:, 2, i + 1] = batch_state[:, 2, i] - (WEBOTS_STEP_TIME / SCALE) * torch.sin(
                batch_state[:, 3, i]) * self.actions[:, i, 0]
            batch_state[:, 3, i + 1] = batch_state[:, 3, i] + (WEBOTS_STEP_TIME / SCALE) * self.actions[:, i, 1] * \
                                       self.actions[:, i, 0] / self.wheelbase
        # ## output shape: [BATCH, HRZ, 4]
        return batch_state.permute(0, 2, 1)

    def is_safe(self, ped_pos, ped_ori):
        # Simple personal space model with an ellipse of radii "a" & "b" and offset by "shift"
        a = 0.8
        b = 1.5
        A = - ped_ori[1]
        shift = 1
        k = shift * torch.cos(A)
        h = - shift * torch.sin(A)
        first_term = torch.square(
            (self.state[:, :, 0] - ped_pos[:, :, 0] - h) * torch.cos(A) + (self.state[:, :, 2] - ped_pos[:, :, 2] - k) * torch.sin(
                A)) / a ** 2
        second_term = torch.square(
            (self.state[:, :, 0] - ped_pos[:, :, 0] - h) * torch.sin(A) - (self.state[:, :, 2] - ped_pos[:, :, 2] - k) * torch.cos(
                A)) / b ** 2
        check = (first_term + second_term) < 1
        return check.int()

    def reset(self):
        # self.simulationReset()
        for ped in self.peds:
            ped.restartController()
        # add2pickle("Herdr_data_train.pkl", self.df)
        # add2pickle("State_rewards.pkl", self.stateR_df, overwrite=True)
        self.logger.restartController()
        self.hircus.restartController()
        # new_goal()
        pass

    def checkreset(self):
        pos = self.hircus.getPosition()
        # if np.sqrt(pos[0] ** 2 + pos[2] ** 2) >= 9.5:
        #     self.simulationReset()
        #     self.reset()
        # if self.getTime() > 50:
        #     self.reset()
        dist2goal = np.sqrt((pos[0] - GOAL[0, 0, 0]) ** 2 + (pos[2] - GOAL[0, 0, 1]) ** 2)
        if dist2goal < 0.75:
            # if within 0.5 [m] of goal
            print("Made it!!")
            self.reset()

    def todataset(self):
        # if self.recognize() and self.train:
        self.now = datetime.now()
        group = file.create_group(f"{self.now}")
        d1 = file.create_dataset(f"{self.now}/actions", shape=self.actions.shape)
        d1[...] = self.actions.detach().numpy()
        d2 = file.create_dataset(f"{self.now}/gnd", shape=self.event.shape)
        d2[...] = self.event.detach().numpy()
        maxlen = len(str(self.now))
        dtipe = 'S{0}'.format(maxlen)
        d3 = file.create_dataset(f"{self.now}/img", (BATCH,), dtype=dtipe)
        d3[...] = f"{self.now}.jpg"
        # d3 = file.create_dataset(f"{self.now}/img", data=np.uint8(self.frame[0].numpy()))

        # for i, sample in enumerate(self.actions):
        #     in2df = pd.DataFrame([Ped_sample(self.actions[i, :, :].detach(),
        #                                      self.event[i, :].detach(), f"{self.now}.png")])
        #     self.df = self.df.append(in2df, ignore_index=True)
        # save_image(self.frame[0]/255, f'./images/{self.now}.jpg')
        cv2.imwrite(f'./images/{self.now}.jpg', self.frame[0].permute(1, 2, 0).numpy())

    def Herdr(self):
        frame = np.asarray(np.frombuffer(self.camera.getImage(), dtype=np.uint8))
        frame = np.reshape(np.ravel(frame), (self.height, self.width, 4), order='C')
        frame = torch.tensor(frame[:, :, 0:3]).float()
        self.frame = frame.permute(2, 0, 1).unsqueeze(0)
        self.frame = self.frame.repeat(BATCH, 1, 1, 1)
        self.actions = self.planner.sample_new(batches=BATCH)
        r = self.reward()
        best_r_arg = torch.argmin(torch.sum(r, dim=0))
        # best_r_arg = torch.randint(0, BATCH, (1, 1)).item()

        # update motors and check for nan values
        # self.update_motors(float(self.actions[best_r_arg, 0, 0]), float(self.actions[best_r_arg, 0, 1]))

        # Save To DataSet
        self.todataset()

        # Update action mean and check for reset
        r = - r
        self.planner.update_new(r, self.actions)
        self.update_motors(float(self.planner.mean[0, 0]), float(self.planner.mean[1, 0]))
        self.checkreset()


"""Set the inference method and goal position."""
opt_parser = optparse.OptionParser()
opt_parser.add_option("--training", action="store_true", default=False, help="Enable trainer or model (default)")
opt_parser.add_option("--ultrasound", action="store_true", default=False, help="Enable use of ultrasound sensor to get ped positon")
opt_parser.add_option("--cmpstk", action="store_true", default=False, help="Enable NCS2 to Acclerate Model Inference")
opt_parser.add_option("--goal", help="Specify Target Position - Format x,z")
options, args = opt_parser.parse_args()

WHEEL_RADIUS = 0.16  # m
WEBOTS_STEP_TIME = 200
DEVICE_SAMPLE_TIME = int(WEBOTS_STEP_TIME * 2)
SCALE = 1000
GNSS_RATE = 1
HRZ = 10
BATCH = 50
if options.goal is None:
    GOAL = torch.tensor([uniform(-6, 3), uniform(-6, 6)]).repeat(BATCH, HRZ, 1)
else:
    str_list = options.goal.split(",")
    goal_list = []
    for c in str_list:
        goal_list.append(float(c))
    GOAL = torch.tensor(goal_list).repeat(BATCH, HRZ, 1)

print(f"X: {GOAL[0,0,0]:.4f}, Z: {GOAL[0, 0, 1]:.4f}")
WEBOTS_ROBOT_NAME = "CapraHircus"
Ped_sample = make_dataclass("Sample", [("Actions", float), ("Ground_Truth", float), ("Image_Name", str)])
State_Event = make_dataclass("States", [("State", float), ("Event_Prob", float), ("Target_Pos", float)])

"""####### FOR NCS2 ######"""
if options.cmpstk:
    from openvino.inference_engine import IECore, IENetwork, Blob, TensorDesc
    ie = IECore()
    net = IECore.read_network(ie, 'Herdr.xml', 'Herdr.bin')
    output_blob = next(iter(net.outputs))
    exec_net = ie.load_network(network=net, device_name='MYRIAD', num_requests=1)
    inference_request = exec_net.requests[0]

controller = Hircus(train=options.training, accel=options.cmpstk, ultra=options.ultrasound)
fig = plt.figure(figsize=(16, 8.9), dpi=80)
cmap = mpl.cm.magma
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Probability')
file = h5py.File(f'hdf5s/{datetime.now()}.h5', 'w')
# writer = animation.FFMpegWriter(fps=5)
# writer.setup(fig, 'actions_cam_view.mp4')
while not controller.step(WEBOTS_STEP_TIME) == -1:
    controller.Herdr()
    # controller.customdata.setSFString(str(torch.sum(controller.event).item()))
    # writer.grab_frame()
file.close()
# writer.finish()


