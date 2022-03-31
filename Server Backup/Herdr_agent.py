from cgitb import handler
import glob
import imp
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import faulthandler
import carla
import cv2
import time
import numpy as np
import torch
from torch import nn
import time
from datetime import datetime

from Badgrnet import HERDR
from actionplanner import HERDRPlan
from metrics_utils import plot_action_cam_view, plot_actions, plot_trajectory


def location2tensor(location):
    ls = [location.x, location.y, location.z]
    return torch.tensor(ls)

class Herdragent():
    actor_list = []
    sensor_list = []
    collision_hist = []
    pos_hist = []
    action_hist = []
    im_hist = []
    GND_hist = []
    im_width = 640 # pixels
    im_height = 480 # pixels
    FOV = 150.0 # degrees

    td_im_width = 500 # pixels
    td_im_height = 500 # pixels
    td_FOV = 90.0 # degrees
    
    control_freq = 5 # Hz
    horizon = 2*control_freq
    batches = 70
    init_vel = 1.5 # m/s
    safety_gain = 1.1 # magic number 
    wheelbase = 0.7 # m
    vehicle = None
    CAM_SHOW = False # bool to show front rgb camera preview
    frame = None # initalizer for front rgb camera img
    depth_frame = None # initalizer for front depth camera img
    topview = None # initalizer for top view rgb camera img
    done = False # flag for done
    success = False
    # front_wheel = carla.VehicleWheelLocation.Front_Wheel
    score = None
    first_call = False
    p2pdist = 0

    def __init__(self, training=False, model_name=None):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(8.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.omafiets = self.blueprint_library.filter('omafiets')[0]
        self.planner = HERDRPlan(Horizon=self.horizon, vel_init=self.init_vel)

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print("Use GPU")
        else:
            self.device = torch.device('cpu')
            print("Use CPU")

        if training:
            if model_name == None:
                self.model = HERDR(Horizon=self.horizon)
            else:
                self.model = torch.load(f'/home/nathan/HERDR/models/{model_name}')
            self.model.eval()
            self.model.to(self.device)
            self.sig = nn.Sigmoid()
            self.infer = self.calc_score_training
        else:
            # self.model = torch.load('/home/nathan/HERDR/models/Herdr_cross22-02-2022--17:42:02.pth')
            self.model = torch.load(f'/home/nathan/HERDR/models/{model_name}')
            self.model.model_out = nn.Sequential(
                        self.model.model_out,
                        nn.Sigmoid())
            self.model.eval()
            self.model.to(self.device)
            self.infer = self.calc_score_model
    
    def get_goal(self, tf):
        trans = carla.Transform()
        trans.location = self.world.get_random_location_from_navigation()
        # trans.location = carla.Location(x=125., y=50., z=0.16)
        ''' *** Try to remove .repeat() to save memory *** '''
        self.GOAL = torch.tensor([trans.location.x, trans.location.y]).repeat(self.batches, 1, 1)
        print(f'Current location is: {tf.location}')
        print(f'Goal location is: {trans.location}')

    def img_process(self, image, cam):
        img = torch.tensor(image.raw_data)
        if cam == 'car':
            img = img.reshape((self.im_height, self.im_width, 4))
            img = img[:, :, :3]
            '''Image shape [im_height, im_width, 3], Torch tensor BGR'''
            if self.CAM_SHOW == True:
                cv2.imshow("Front_CAM", img.float().numpy()/255)
                # cv2.waitKey(1)
            '''Image shape [3, im_height, im_width]'''
            self.frame = img.permute(2, 0, 1).float()
        elif cam == 'td':
            img = img.reshape((self.td_im_height, self.td_im_width, 4))
            img = img[:, :, :3]
            '''Image shape [im_height, im_width, 3], Torch tensor BGR'''
            self.topview = img.float()
                
    def collison_check(self, event):
        if not self.done:
            self.collision_hist.append(event.timestamp)
    
    def lane_check(self, event):
        if not self.done:
            print("Drove onto Road")
            self.collision_hist.append(event.timestamp)
    
    def reset(self):
        self.pos_hist = []
        self.done = False

        while self.vehicle is None:
            transform = carla.Transform(self.world.get_random_location_from_navigation())
            # transform = carla.Transform(carla.Location(x=115.203445, y=-37.268463, z=0.177637))
            self.vehicle = self.world.try_spawn_actor(self.omafiets, transform)
        self.actor_list.append(self.vehicle)

        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('sensor_tick', f'{1/self.control_freq}')

        '''Add forward facing camera '''
        self.camera_bp.set_attribute("image_size_x", f'{self.im_width}')
        self.camera_bp.set_attribute("image_size_y", f'{self.im_height}')
        self.camera_bp.set_attribute("fov", f'{self.FOV}')
        car_camera_transform = carla.Transform(carla.Location(x=0.5, z=1.4))
        self.cam = self.world.spawn_actor(self.camera_bp, car_camera_transform, attach_to=self.vehicle)
        self.sensor_list.append(self.cam)
        self.cam.listen(lambda image: self.img_process(image, 'car'))

        '''Add top down camera '''
        td_cam_pos = self.vehicle.get_location()
        topdown_camera_transform = carla.Transform(carla.Location(x=td_cam_pos.x, y=td_cam_pos.y, z=15), carla.Rotation(pitch=-90))
        self.camera_bp.set_attribute("image_size_x", f'{self.td_im_width}')
        self.camera_bp.set_attribute("image_size_y", f'{self.td_im_height}')
        self.camera_bp.set_attribute("fov", f'{self.td_FOV}')
        self.tdcam = self.world.spawn_actor(self.camera_bp, topdown_camera_transform)
        self.sensor_list.append(self.tdcam)
        self.tdcam.listen(lambda image: self.img_process(image, 'td'))

        ''' Add collision sensor '''
        collison_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collison_sensor = self.world.spawn_actor(collison_sensor, car_camera_transform, attach_to=self.vehicle)
        self.sensor_list.append(self.collison_sensor)
        self.collison_sensor.listen(lambda event: self.collison_check(event))

        ''' Add road detection sensor '''
        lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_sensor, car_camera_transform, attach_to=self.vehicle)
        self.sensor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self.lane_check(event))

        # del self.camera_bp 
        # del self.blueprint_library
        while self.frame is None:
            self.world.tick()
            time.sleep(0.01)
        self.get_goal(transform)
    
    def update_controls(self):
        # vel = self.vehicle.get_velocity()
        # speed = np.sqrt(vel.x ** 2 + vel.y **2)
        speed = self.vehicle.get_velocity().length()
        if self.planner.mean[0, 0] > speed:
            a = 1.0
        else:
            a = 0.
        self.vehicle.apply_control(carla.VehicleControl(throttle=a, steer=-self.planner.mean[1, 0].item()))

    def calc_score_model(self):
        self.rot = torch.tensor(self.vehicle.get_transform().rotation.yaw).unsqueeze(0)/180*np.pi
        self.state = self.calculate_position()
        ''' goalReward Shape: [BATCH, HRZ] '''
        goalReward = torch.linalg.norm(self.state[:,:,:2]-self.GOAL, dim=2)
        ''' Call Model '''
        img = self.frame.repeat(self.batches, 1, 1, 1)  # .unsqueeze(0)
        indices = torch.tensor([2, 1, 0])
        img = torch.index_select(img, 1, indices)
        self.event = self.model(img.to(self.device), self.actions.to(self.device))[:, :, 0].detach().cpu()
        ''' Scale model output to macth distance for score '''
        event_gain = goalReward.mean()*self.safety_gain
        self.score = goalReward + event_gain * self.event

        return self.score

    def is_safe(self, ped_list):
        self.safe = torch.zeros((1))
        if len(self.collision_hist) > 0:
            self.safe = 1 # which means unsafe
            return
        self.vehicle_location = location2tensor(self.vehicle.get_location())
        for walker in ped_list:
            walker_trans = walker.get_transform()
            walker_pos = location2tensor(walker_trans.location)
            ''' Simple personal space model with an ellipse of radii "a" & "b" and offset by "shift" '''
            a = 0.8
            b = 1.5
            A = torch.tensor(walker_trans.rotation.yaw/180*np.pi)
            shift = 1
            k = shift * torch.cos(A)
            h = - shift * torch.sin(A)
            first_term = torch.square(
                (self.vehicle_location[0] - walker_pos[0] - h) * torch.cos(A) + 
                (self.vehicle_location[1] - walker_pos[1] - k) * torch.sin(A)) / a ** 2
            second_term = torch.square(
                (self.vehicle_location[0] - walker_pos[0] - h) * torch.sin(A) - 
                (self.vehicle_location[1] - walker_pos[1] - k) * torch.cos(A)) / b ** 2
            check = (first_term + second_term) < 1
            check = check.int()
            self.safe = torch.logical_or(check, self.safe).float()
            if self.safe.item() == 1:
                print(f'In Pedestrain Space.')
                return
        return

    def calc_score_training(self):
        self.rot = torch.tensor(self.vehicle.get_transform().rotation.yaw).unsqueeze(0)/180*np.pi
        self.state = self.calculate_position()
        ''' goalReward Shape: [BATCH, HRZ] '''
        goalReward = torch.linalg.norm(self.state[:,:,:2]-self.GOAL, dim=2)
        ''' Call Model '''
        img = self.frame.repeat(self.batches, 1, 1, 1) #.unsqueeze(0)
        indices = torch.tensor([2, 1, 0])
        img = torch.index_select(img, 1, indices)
        self.event = self.sig(self.model(img.to(self.device), self.actions.to(self.device))[:, :, 0].detach().cpu())
        ''' Scale model output to macth distance for score '''
        event_gain = goalReward.mean()*self.safety_gain
        self.score = goalReward + event_gain * self.event
        return self.score

    def calculate_position(self):
        new_pos = location2tensor(self.vehicle.get_transform().location)
        new_state = torch.cat((new_pos, self.rot))
        batch_state = new_state.repeat(self.batches, self.horizon, 1).transpose(1, 2)
        ''' [X Y Z Phi] '''
        for i in range(0, self.horizon - 1):
            batch_state[:, 0, i + 1] = batch_state[:, 0, i] + (1/self.control_freq) * torch.cos(
                batch_state[:, 3, i]) * self.actions[:, i, 0]
            batch_state[:, 1, i + 1] = batch_state[:, 1, i] + (1/self.control_freq) * torch.sin(
                batch_state[:, 3, i]) * self.actions[:, i, 0]
            batch_state[:, 3, i + 1] = batch_state[:, 3, i] - (1/self.control_freq) * self.actions[:, i, 1] * \
                                       self.actions[:, i, 0] / self.wheelbase
        ''' Output shape: [BATCH, HRZ, 4] '''
        return batch_state.permute(0, 2, 1)

    def calc_p2pdist(self, loc):
        loc = location2tensor(loc)
        self.p2pdist = torch.linalg.norm(loc-self.GOAL[0], dim=2)
        
    def reset_check(self):
        pos = location2tensor(self.vehicle.get_location())
        pos = pos[0:2]
        dist2goal = torch.linalg.norm(pos - self.GOAL[0])
        if dist2goal <= 0.75:
            self.done = True
            self.success = True
            print('Made it!!!')

    def step(self):
        self.actions = self.planner.sample_new(batches=self.batches)
        score = self.infer()
        score = - score
        self.planner.update_new(score, self.actions)
        self.action_hist.append(self.planner.mean[:, 0].numpy())
        now = datetime.now()
        self.im_hist.append(f'{now}')
        cv2.imwrite(f'./carla_images/{now}.jpg', self.frame.permute(1, 2, 0).numpy())
        self.update_controls()
        self.reset_check()
        position = location2tensor(self.vehicle.get_location()).numpy()
        if (position[0] != 0) and (position[1] != 0):
            self.pos_hist.append(position)

    def cleanup(self):
        print('Destroying Agent')
        self.client.apply_batch([carla.command.DestroyActor(x.id) for x in self.actor_list])
        print('Destroying Sensors')
        [x.stop() for x in self.sensor_list]
        self.client.apply_batch([carla.command.DestroyActor(x.id) for x in self.sensor_list])
        self.vehicle = None
        self.actor_list.clear()
        self.sensor_list.clear()
        self.collision_hist.clear()
        self.pos_hist.clear()
        self.action_hist.clear()
        self.im_hist.clear()
        self.GND_hist.clear()


if __name__=='__main__':
    print('This is only a class')
    pass