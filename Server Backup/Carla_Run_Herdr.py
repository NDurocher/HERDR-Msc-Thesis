#!/usr/bin/env python

from cgitb import handler
import glob
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
import random
import time
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import h5py
import time
from datetime import datetime
from multiprocessing import Process, Queue
from Herdr_agent import Herdragent

from Badgrnet import HERDR
from actionplanner import HERDRPlan
from metrics_utils import plot_action_cam_view, plot_actions, plot_trajectory

def location2tensor(location):
    ls = [location.x, location.y, location.z]
    return torch.tensor(ls)


class HERDRenv():
    actor_list = []
    controller_list = []
    # sensor_list = []
    collision_hist = []
    pos_hist = []
    action_hist = []
    im_hist = []
    GND_hist = []
    # im_width = 640 # pixels
    # im_height = 480 # pixels
    # FOV = 150.0 # degrees

    # td_im_width = 500 # pixels
    # td_im_height = 500 # pixels
    # td_FOV = 90.0 # degrees
    
    control_freq = 5 # Hz
    # horizon = 2*control_freq
    # batches = 70
    # init_vel = 1.5 # m/s
    # safety_gain = 0.99 # magic number 
    # wheelbase = 0.7 # m
    # vehicle = None
    # CAM_SHOW = False # bool to show front rgb camera preview
    # frame = None # initalizer for front rgb camera img
    # depth_frame = None # initalizer for front depth camera img
    # topview = None # initalizer for top view rgb camera img
    # done = False # flag for done
    # SPL_list = [] 
    # # front_wheel = carla.VehicleWheelLocation.Front_Wheel
    # score = None
    # first_call = False

    plot_hist_front = Queue()
    plot_hist_top = Queue()

    H5File_name = None

    def __init__(self, training=False):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(8.0)
        self.world = self.client.get_world()
        # self.wmap = self.world.get_map()
        new_settings = self.world.get_settings()
        # new_settings.synchronous_mode = True
        # new_settings.max_substeps = 20
        new_settings.fixed_delta_seconds = 1/self.control_freq
        self.world.apply_settings(new_settings) 

        self.blueprint_library = self.world.get_blueprint_library()

        # self.omafiets = self.blueprint_library.filter('omafiets')[0]
        # self.planner = HERDRPlan(Horizon=self.horizon, vel_init=self.init_vel)
        # if training:
        #     self.infer = self.calc_score_training
        # else:
        #     self.model = torch.load('/home/nathan/HERDR/models/env_cross22-02-2022--17:42:02.pth')
        #     self.model.model_out = nn.Sequential(
        #                 self.model.model_out,
        #                 nn.Sigmoid())
        #     self.model.eval()
        #     self.model.to(device)
        #     self.infer = self.calc_score_model

    # def get_goal(self):
    #     trans = carla.Transform()
    #     # trans.location = self.world.get_random_location_from_navigation()
    #     trans.location = carla.Location(x=125., y=50., z=0.16)
    #     ''' *** Try to remove .repeat() to save memory *** '''
    #     self.GOAL = torch.tensor([trans.location.x, trans.location.y]).repeat(self.batches, 1, 1)
    #     print(f'Current location is: {self.vehicle.get_location()}')
    #     print(f'Goal location is: {trans.location}')
    
    def reworld(self):
        self.client.reload_world()
        self.world.wait_for_tick()

    # def img_process(self, image, cam):
    #     img = torch.tensor(image.raw_data)
    #     if cam == 'car':
    #         img = img.reshape((self.im_height, self.im_width, 4))
    #         img = img[:, :, :3]
    #         '''Image shape [im_height, im_width, 3], Torch tensor BGR'''
    #         if self.CAM_SHOW == True:
    #             cv2.imshow("Front_CAM", img.float().numpy()/255)
    #             cv2.waitKey(1)
    #         '''Image shape [3, im_height, im_width]'''
    #         self.frame = img.permute(2, 0, 1).float()
    #     elif cam == 'td':
    #         img = img.reshape((self.td_im_height, self.td_im_width, 4))
    #         img = img[:, :, :3]
    #         '''Image shape [im_height, im_width, 3], Torch tensor BGR'''
    #         self.topview = img.float()
                
    # def collison_check(self, event):
    #     if not self.done:
    #         self.collision_hist.append(event.timestamp)
        
    def pop_map_pedestrians(self, num_peds=200):
        # self.ped_actor_start_idx = len(self.actor_list)-1
        controller_bp = self.blueprint_library.find('controller.ai.walker')
        for i in range(num_peds):
            walker_bp = random.choice(self.blueprint_library.filter('walker.pedestrian.*'))
            trans = carla.Transform()
            trans.location = self.world.get_random_location_from_navigation()
            trans.location.z += 1

            walker_actor = self.world.try_spawn_actor(walker_bp, trans)
            self.world.wait_for_tick()
            if walker_actor is None:
                continue
            self.actor_list.append(walker_actor)

            controller_walker = self.world.try_spawn_actor(controller_bp, carla.Transform(), walker_actor)
            self.world.wait_for_tick()

            controller_walker.start()
            controller_walker.go_to_location(self.world.get_random_location_from_navigation())

            self.controller_list.append(controller_walker)
        # self.ped_actor_stop_idx = len(self.controller_list)-1
        del self.blueprint_library
        print('Pedestrians Added.')

    def reset(self):
        # self.actor_list = []
        # self.controller_list = []
        # self.sensor_list = []
        # self.pos_hist = []
        # self.done = False

        # while self.vehicle is None:
        #     transform = carla.Transform(self.world.get_random_location_from_navigation())
        #     self.vehicle = self.world.try_spawn_actor(self.omafiets, transform)
        # self.actor_list.append(self.vehicle)

        # self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        # self.camera_bp.set_attribute('sensor_tick', f'{1/self.control_freq}')

        # '''Add forward facing camera '''
        # self.camera_bp.set_attribute("image_size_x", f'{self.im_width}')
        # self.camera_bp.set_attribute("image_size_y", f'{self.im_height}')
        # self.camera_bp.set_attribute("fov", f'{self.FOV}')
        # car_camera_transform = carla.Transform(carla.Location(x=0.5, z=1.4))
        # self.cam = self.world.spawn_actor(self.camera_bp, car_camera_transform, attach_to=self.vehicle)
        # self.sensor_list.append(self.cam)
        # self.cam.listen(lambda image: self.img_process(image, 'car'))

        # '''Add top down camera '''
        # td_cam_pos = self.vehicle.get_location()
        # topdown_camera_transform = carla.Transform(carla.Location(x=td_cam_pos.x, y=td_cam_pos.y, z=15), carla.Rotation(pitch=-90))
        # self.camera_bp.set_attribute("image_size_x", f'{self.td_im_width}')
        # self.camera_bp.set_attribute("image_size_y", f'{self.td_im_height}')
        # self.camera_bp.set_attribute("fov", f'{self.td_FOV}')
        # self.tdcam = self.world.spawn_actor(self.camera_bp, topdown_camera_transform)
        # self.sensor_list.append(self.tdcam)
        # self.tdcam.listen(lambda image: self.img_process(image, 'td'))

        # ''' Add collision sensor '''
        # collison_sensor = self.blueprint_library.find("sensor.other.collision")
        # self.collison_sensor = self.world.spawn_actor(collison_sensor, car_camera_transform, attach_to=self.vehicle)
        # self.sensor_list.append(self.collison_sensor)
        # self.collison_sensor.listen(lambda event: self.collison_check(event))

        # self.get_goal()
        self.H5File = h5py.File(f'/home/nathan/HERDR/carla_hdf5s/{self.H5File_name}.h5', 'a')
        # self.GND_hist = []
        # self.action_hist = []
        # self.im_hist = []
        # self.plot_hist_front = Queue()
        # self.plot_hist_top = Queue()
        # self.collision_hist = []

        # del self.camera_bp 
        # while self.frame is None:
        #     time.sleep(0.01)

    # def update_controls(self):
    #     # vel = self.vehicle.get_velocity()
    #     # speed = np.sqrt(vel.x ** 2 + vel.y **2)
    #     speed = self.vehicle.get_velocity().length()
    #     if self.planner.mean[0, 0] > speed:
    #         a = 1.0
    #     else:
    #         a = 0.
    #     self.vehicle.apply_control(carla.VehicleControl(throttle=a, steer=-self.planner.mean[1, 0].item()))

    # def calc_score_model(self):
    #     self.rot = torch.tensor(self.vehicle.get_transform().rotation.yaw).unsqueeze(0)/180*np.pi
    #     self.state = self.calculate_position()
    #     ''' goalReward Shape: [BATCH, HRZ] '''
    #     goalReward = torch.linalg.norm(self.state[:,:,:2]-self.GOAL, dim=2)
    #     ''' Call Model '''
    #     img = self.frame.unsqueeze(0) #.repeat(self.batches, 1, 1, 1)
    #     indices = torch.tensor([2, 1, 0])
    #     img = torch.index_select(img, 1, indices)
    #     self.event = self.model(img.to(device), self.actions.to(device))[:, :, 0].detach().cpu()
    #     ''' Scale model output to macth distance for score '''
    #     event_gain = goalReward.mean()*self.safety_gain
    #     self.score = goalReward + event_gain * self.event

    #     return self.score

    # def is_safe(self):
    #     self.safe = torch.zeros((1))
    #     if len(self.collision_hist) > 0:
    #         self.safe = 1 # which means unsafe
    #         return
    #     self.vehicle_location = location2tensor(self.vehicle.get_location())
    #     for walker in self.controller_list:
    #         walker_trans = walker.get_transform()
    #         walker_pos = location2tensor(walker_trans.location)
    #         ''' Simple personal space model with an ellipse of radii "a" & "b" and offset by "shift" '''
    #         a = 0.8
    #         b = 1.5
    #         A = torch.tensor(walker_trans.rotation.yaw/180*np.pi)
    #         shift = 1
    #         k = shift * torch.cos(A)
    #         h = - shift * torch.sin(A)
    #         first_term = torch.square(
    #             (self.vehicle_location[0] - walker_pos[0] - h) * torch.cos(A) + 
    #             (self.vehicle_location[1] - walker_pos[1] - k) * torch.sin(A)) / a ** 2
    #         second_term = torch.square(
    #             (self.vehicle_location[0] - walker_pos[0] - h) * torch.sin(A) - 
    #             (self.vehicle_location[1] - walker_pos[1] - k) * torch.cos(A)) / b ** 2
    #         check = (first_term + second_term) < 1
    #         check = check.int()
    #         self.safe = torch.logical_or(check, self.safe).float()
    #         if self.safe.item() == 1:
    #             return
    #     return

    # def calc_score_training(self):
    #     start_idx = self.ped_actor_stop_idx
    #     stop_idx = self.ped_actor_stop_idx
    #     self.rot = torch.tensor(self.vehicle.get_transform().rotation.yaw).unsqueeze(0)/180*np.pi
    #     self.state = self.calculate_position()
    #     goalReward = torch.linalg.norm(self.state[:,:,:2]-self.GOAL, dim=2)
    #     self.event = torch.zeros((self.batches, self.horizon))
    #     for walker in self.controller_list:
    #         try:
    #             walker_trans = walker.get_transform()
    #         except:
    #             walker_trans = walker
    #         # print(walker_trans)
    #         walker_pos = location2tensor(walker_trans.location)
    #         # Simple personal space model with an ellipse of radii "a" & "b" and offset by "shift"
    #         # a = 0.8
    #         # b = 1.5
    #         # A = torch.tensor(walker_trans.rotation.yaw/180*np.pi)
    #         # shift = 1
    #         # k = shift * torch.cos(A)
    #         # h = - shift * torch.sin(A)
    #         # first_term = torch.square(
    #         #     (self.state[:, :, 0] - walker_pos[0] - h) * torch.cos(A) + (self.state[:, :, 1] - walker_pos[1] - k) * torch.sin(
    #         #         A)) / a ** 2
    #         # second_term = torch.square(
    #         #     (self.state[:, :, 0] - walker_pos[0] - h) * torch.sin(A) - (self.state[:, :, 1] - walker_pos[1] - k) * torch.cos(
    #         #         A)) / b ** 2
    #         # check = (first_term + second_term) < 1
    #         ''' Try making just a circle radius around the objects because nothing is being detected '''
    #         radius = (self.state[:, :, 0] - walker_pos[0]) ** 2 + (self.state[:, :, 1] - walker_pos[1]) ** 2
    #         check = radius < 3 ** 2
    #         check = check.int()
    #         self.event = torch.logical_or(check, self.event).float()
    #     if len(self.controller_list) > stop_idx + 20:
    #         self.controller_list = self.controller_list[:stop_idx]
    #     event_gain = goalReward.mean()*self.safety_gain
    #     self.score = goalReward + event_gain * self.event
    #     return self.score

    # def calculate_position(self):
    #     new_pos = location2tensor(self.vehicle.get_transform().location)
    #     new_state = torch.cat((new_pos, self.rot))
    #     batch_state = new_state.repeat(self.batches, self.horizon, 1).transpose(1, 2)
    #     ''' [X Y Z Phi] '''
    #     for i in range(0, self.horizon - 1):
    #         batch_state[:, 0, i + 1] = batch_state[:, 0, i] + (1/self.control_freq) * torch.cos(
    #             batch_state[:, 3, i]) * self.actions[:, i, 0]
    #         batch_state[:, 1, i + 1] = batch_state[:, 1, i] + (1/self.control_freq) * torch.sin(
    #             batch_state[:, 3, i]) * self.actions[:, i, 0]
    #         batch_state[:, 3, i + 1] = batch_state[:, 3, i] - (1/self.control_freq) * self.actions[:, i, 1] * \
    #                                    self.actions[:, i, 0] / self.wheelbase
    #     ''' Output shape: [BATCH, HRZ, 4] '''
    #     return batch_state.permute(0, 2, 1)
    
    # def reset_check(self):
    #     pos = location2tensor(self.vehicle.get_location())
    #     pos = pos[0:2]
    #     dist2goal = torch.linalg.norm(pos - self.GOAL[0])
    #     if dist2goal <= 0.75:
    #         self.done = True
    #         print('Made it!!!')

    # def step(self):
    #     self.actions = self.planner.sample_new(batches=self.batches)
    #     score = self.infer()
    #     score = - score
    #     self.planner.update_new(score, self.actions)
    #     self.action_hist.append(self.planner.mean[:, 0].numpy())
    #     now = datetime.now()
    #     self.im_hist.append(f'{now}.jpg')
    #     cv2.imwrite(f'./carla_images/{now}.jpg', self.frame.permute(1, 2, 0).numpy())
    #     self.update_controls()
    #     self.reset_check()
    #     self.pos_hist.append(location2tensor(self.vehicle.get_location()).numpy())

    def set_recordings(self):
        plt.clf()
        fig = plt.figure(figsize=(16, 9), dpi=80)
        cmap = mpl.cm.YlOrRd
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Probability')
        fps = 5
        self.writer = animation.FFMpegWriter(fps=fps)
        self.writer.setup(fig, './actions_cam_view.mp4')
        self.top_writer = animation.FFMpegWriter(fps=fps)
        self.top_writer.setup(fig, './Top_view.mp4')

    def get_recordings(self):
        self.plot_hist_front.put([location2tensor(self.vehicle.get_location()), self.frame.permute(1,2,0), self.event, self.state])
        td_cam_pos = self.vehicle.get_location()
        topdown_camera_transform = carla.Transform(carla.Location(x=td_cam_pos.x, y=td_cam_pos.y, z=15), carla.Rotation(pitch=-90))
        self.tdcam.set_transform(topdown_camera_transform)
        self.plot_hist_top.put([self.state, self.event, location2tensor(topdown_camera_transform.location).numpy(), self.GOAL[0,0,:], self.topview])
        return
        
    def background_save(self, Q1, Q2):
        count = 0
        while(True):
            if(Q1.empty() | Q2.empty()):
                time.sleep(1)
                count += 1
                if count == 5:
                    print('Saving timed out')
                    return
            else:
                count = 0
                front = Q1.get()
                if 'done' in front:
                    print('Saving Done')
                    return
                plot_action_cam_view(*front)
                self.writer.grab_frame()
                top = Q2.get()
                plot_actions(*top)
                self.top_writer.grab_frame()

    def new_run(self, action_hist, im_hist, GND_hist):
        self.H5File = h5py.File(f'/home/nathan/HERDR/carla_hdf5s/{self.H5File_name}.h5', 'a')
        self.now = datetime.now()
        group = self.H5File.create_group(f"{self.now}")
        self.d1 = self.H5File.create_dataset(f"{self.now}/actions", data=action_hist)
        self.d2 = self.H5File.create_dataset(f"{self.now}/gnd", data=GND_hist)
        maxlen = len(str(self.now))
        dtipe = 'S{0}'.format(maxlen)
        self.d3 = self.H5File.create_dataset(f"{self.now}/img", data=im_hist, dtype=dtipe)

    def cleanup(self):
        self.actor_list.clear()
        self.controller_list.clear()

def main():
    # while True:
    for j in range(1):
        env = HERDRenv()
        H5File_name = f'{datetime.now()}'
        # faulthandler.enable()
        env.H5File_name = H5File_name
        env.reworld()
        env.reset()
        env.pop_map_pedestrians(num_peds=200)
        try:
            for i in range(10):
            # if (i % 3) == 0:
                # H5File_name = f'{datetime.now()}'
                Herdr = Herdragent(training=False)
                Herdr.reset()
                # env.set_recordings()
                # job = Process(target=env.background_save, args=(env.plot_hist_front,env.plot_hist_top,), daemon=True)
                # job.start()
                print(f'--- START Run {i+1} ---')
                start_time = time.time()
                save_frame_count = 0
                while not Herdr.done | (time.time() - start_time >= 600):
                    Herdr.step()
                    # if save_frame_count % 3 == 0:
                    #     env.get_recordings()
                    env.world.wait_for_tick()
                    Herdr.is_safe(env.controller_list)
                    Herdr.GND_hist.append(Herdr.safe)
                    if len(Herdr.collision_hist) > 0:
                        if Herdr.GND_hist[-1] == 0:
                            Herdr.GND_hist[-1] = 1
                        break
                    save_frame_count += 1
                    if save_frame_count % 500 == 0:
                        print(f"PING - I'm Alive -")
                    # time.sleep(0.1)
                    env.world.tick()

                Herdr.done = True
                # env.plot_hist_front.put('done')
                print(f"--- DONE Run {i+1} ---")
                print(f"Sim Time est: {int(save_frame_count/5)} seconds")
                # job.join()
                # job.close()
                # start_step_time = time.time()
                env.new_run(Herdr.action_hist, Herdr.im_hist, Herdr.GND_hist)
                env.H5File.close()
                # env.top_writer.finish()
                # env.writer.finish()
                # plot_trajectory(np.asarray(env.pos_hist), torch.ones((len(env.pos_hist))), env.GOAL[0,0,:], collision=len(env.collision_hist))
                # plt.savefig('./trajectory.png')
                # plt.close('all')
                Herdr.cleanup()
                del Herdr
                print(f'Real time: {time.time()-start_time:.4f}')

        finally:
            print('Destroying Pedestrians')
            env.client.apply_batch([carla.command.DestroyActor(x.id) for x in env.actor_list])
            # print('Destroying Sensors')
            # [x.stop() for x in env.sensor_list]
            # env.client.apply_batch([carla.command.DestroyActor(x.id) for x in env.sensor_list])
            print('Destroying AI Controllers')
            [x.stop() for x in env.controller_list]
            env.client.apply_batch([carla.command.DestroyActor(x.id) for x in env.controller_list])
            print('Done.')

        env.cleanup()
        del env


if __name__ == '__main__':
    main()