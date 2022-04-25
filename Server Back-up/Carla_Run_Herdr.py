#!/usr/bin/env python

from cgi import test
from cgitb import handler
import glob
import os
from secrets import choice
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
import subprocess
import cv2
import csv
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
import pickle
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

from Badgrnet import HERDR
from Herdr_agent import Herdragent
from Ped_agent import Pedagent
from actionplanner import HERDRPlan
from Carla_Trainer import carla_hdf5dataclass
from metrics_utils import plot_action_cam_view, plot_actions, plot_trajectory, count_data_ratio

def location2tensor(location):
    ls = [location.x, location.y, location.z]
    return torch.tensor(ls)

def EucDistance(x,y):
    if type(x) == carla.libcarla.Location:
        x = location2tensor(x)
    if type(y) == carla.libcarla.Location:
        y = location2tensor(y)
    if type(x) == type(np.array([])):
        x = torch.tensor(x)
    if type(y) == type(np.array([])):
        y = torch.tensor(y)
    dist = torch.norm(x[0:2]-y[0:2])
    return dist

class HERDRenv():
    actor_list = []
    controller_list = []
    # sensor_list = []
    collision_hist = []
    pos_hist = []
    # ped_distance_hist = []
    ped_space_count = 0
    SPL_hist = []
    control_freq = 5 # Hz
    plot_hist_front = Queue()
    plot_hist_top = Queue()
    run = 0
    # weatherparas = ["ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "SoftRainNoon", "MidRainyNoon", "HardRainNoon", 
    #               "ClearSunset", "CloudySunset", "WetSunset", "WetCloudySunset", "SoftRainSunset", "MidRainSunset", "HardRainSunset"]

    H5File_name = None

    def __init__(self, training=False):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(8.0)
        # try:
        #     self.world = self.client.get_world()
        # except:
        #     bash_result = subprocess.Popen(['sh', '/opt/carla-simulator/CarlaUE4.sh'])
        #     print(f"Carla said: {bash_result}")
        #     time.sleep(1)
        self.world = self.client.get_world()
        # self.wmap = self.world.get_map()
        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.max_substeps = 16
        new_settings.max_substep_delta_time = 0.0125
        new_settings.fixed_delta_seconds = 1/self.control_freq
        self.world.apply_settings(new_settings) 
        self.blueprint_library = self.world.get_blueprint_library()
    
    def reworld(self):
        self.client.reload_world()
        self.world.wait_for_tick()
        
    def pop_map_pedestrians(self, num_peds=200, test_block=None):
        # self.ped_actor_start_idx = len(self.actor_list)-1
        controller_bp = self.blueprint_library.find('controller.ai.walker')
        if test_block != None:
            self.load_spawns(f'/home/nathan/HERDR/spawn_locations_test{test_block}.txt')

        for i in range(num_peds):
            walker_bp = random.choice(self.blueprint_library.filter('walker.pedestrian.*'))
            if test_block is None:
                trans = carla.Transform()
                trans.location = self.world.get_random_location_from_navigation()
                trans.location.z += 1
            else:
                trans = self.get_spawn()

            walker_actor = self.world.try_spawn_actor(walker_bp, trans)
            
            self.world.tick()
            if walker_actor is None:
                continue
            self.actor_list.append(walker_actor)

            controller_walker = self.world.try_spawn_actor(controller_bp, carla.Transform(), walker_actor)
            
            self.world.tick()

            controller_walker.start()
            controller_walker.go_to_location(self.world.get_random_location_from_navigation())

            self.controller_list.append(controller_walker)
        
        # del self.blueprint_library
        print('Pedestrians Added.')

    def reset(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(8.0)
        self.world = self.client.get_world()
        # self.wmap = self.world.get_map()
        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.max_substeps = 16
        new_settings.max_substep_delta_time = 0.0125
        new_settings.fixed_delta_seconds = 1/self.control_freq
        self.world.apply_settings(new_settings) 
        preset_list = [item for item in dir(carla.WeatherParameters)[0:22] if 'Night' not in item]
        dict_WP = carla.WeatherParameters.__dict__
        self.world.set_weather(dict_WP[preset_list[random.randint(0,len(preset_list)-1)]])

    def pathlength(self, pos_list):
        np_pos_hist = np.asarray(pos_list)
        x, y = np_pos_hist[:,0], np_pos_hist[:,1]
        n = len(x)
        lv = [np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) for i in range(1, n)]
        L = sum(lv)
        return L

    def dist2peds(self,agent):
        agent_location = agent.vehicle.get_location()
        dist_list = []
        for ped in self.actor_list:
            ped_location = ped.get_location()
            dist_list.append(EucDistance(ped_location, agent_location))
        dist_list = np.asarray(dist_list) 
        min_dist = dist_list.min()
        avg_dist = dist_list.mean()
        max_dist = dist_list.max()
        return min_dist
    
    def calc_SPL(self):
        sum_var = 0
        for success, length, p2pdist in self.SPL_hist:
            sum_var += success * (p2pdist/np.max([p2pdist,length,1e-9]))
        spl = 1/len(self.SPL_hist)*sum_var
        print(f'\n&& SPL: {spl:.4f} &&')
        return spl
       
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

    def get_recordings(self, agent):
        self.plot_hist_front.put([location2tensor(agent.vehicle.get_location()), agent.frame.permute(1,2,0), agent.event, agent.state, agent.planner.mean])
        td_cam_pos = agent.vehicle.get_location()
        topdown_camera_transform = carla.Transform(carla.Location(x=td_cam_pos.x, y=td_cam_pos.y, z=15), carla.Rotation(pitch=-90))
        agent.tdcam.set_transform(topdown_camera_transform)
        self.plot_hist_top.put([agent.state, agent.event, location2tensor(topdown_camera_transform.location).numpy(), agent.GOAL[0,0,:], agent.topview])
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

    def new_run(self, action_hist, im_hist, GND_hist, folder):
        dir_name = Path(Path.cwd())
        p = Path(f'{dir_name}/all_carla_hdf5s/{folder}')
        if not p.is_dir():
            os.mkdir(f'{dir_name}/all_carla_hdf5s/{folder}')
        self.H5File = h5py.File(f'{dir_name}/all_carla_hdf5s/{folder}/{self.H5File_name}.h5', 'a')
        self.now = datetime.now()
        group = self.H5File.create_group(f"{self.now}")
        self.d1 = self.H5File.create_dataset(f"{self.now}/actions", data=action_hist)
        self.d2 = self.H5File.create_dataset(f"{self.now}/gnd", data=GND_hist)
        maxlen = len(str(self.now))
        dtipe = 'S{0}'.format(maxlen)
        self.d3 = self.H5File.create_dataset(f"{self.now}/img", data=im_hist, dtype=dtipe)

    def cleanup(self):
        print('Destroying Pedestrians')
        self.client.apply_batch([carla.command.DestroyActor(x.id) for x in self.actor_list])
        self.world.tick()
        print('Destroying AI Controllers')
        [x.stop() for x in self.controller_list]
        self.client.apply_batch([carla.command.DestroyActor(x.id) for x in self.controller_list])
        self.world.tick()
        print('Done.')
        self.actor_list.clear()
        self.controller_list.clear()
        self.world.tick()

    def load_spawns(self, file):
        self.spawn_locations = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.spawn_locations.append(row)
            self.spawn_locations = np.asarray(self.spawn_locations).astype(np.int16)

    def get_spawn(self):
        rot = float(np.random.uniform(0,360,1))
        loc = np.random.choice(np.arange(0,self.spawn_locations.shape[0]-1))
        return carla.Transform(carla.Location(x=float(self.spawn_locations[loc,0]), y=float(self.spawn_locations[loc,1]), z=0.177637), carla.Rotation(yaw=rot))

def main():
    # while True:
    faulthandler.enable()
    torch.set_default_dtype(torch.float32)
    max_loss = 10000
    try:
        log_time = datetime.now().strftime("%d-%m-%Y--%H-%M")
        # model_start = '07-04-2022--14:41'
        model_start = 'carla22-04-2022--09:48'
        recording_data = log_time
        env = HERDRenv()
        # env.reworld()
        # env.reset()
        env.pop_map_pedestrians(num_peds=200)
        # writer = SummaryWriter(log_dir=f'/home/nathan/HERDR/carla_logs/{log_time}')
        Herdr = Herdragent(training=False, recording=recording_data, model_name=f'carla{model_start}.pth')  # model_name=None
        if os.path.isfile(f'/home/nathan/HERDR/pickles/training_counts_{log_time}.pkl'):
            with open(f'/home/nathan/HERDR/pickles/training_counts_{log_time}.pkl','rb') as f:
                total_sim_time, env.run, end_step, env.SPL_hist = pickle.load(f)
        else:
            end_step = 0
            total_sim_time = 0
        round = 0

        # for j in range(10):
        while True:
            print(f"Round {round+1}!\n")
            if recording_data:
                H5File_name = f'{datetime.now()}'
                env.H5File_name = H5File_name
            try:
                num_epoch = 100
                for i in range(num_epoch):
                    env.run += 1
                    env.ped_space_count = 0
                    # H5File_name = f'{datetime.now()}'
                    Herdr.reset()
                    # env.set_recordings()
                    # job = Process(target=env.background_save, args=(env.plot_hist_front,env.plot_hist_top,), daemon=True)
                    # job.start()
                    print(f'--- START Run {i+1}/Round {round+1} at: {datetime.now().strftime("%d-%m-%Y--%H:%M:%S")} ---')
                    start_time = time.time()
                    save_frame_count = 0
                    while not Herdr.done | (time.time() - start_time >= 300):
                        Herdr.step()
                        # if save_frame_count % 3 == 0:
                        #     env.get_recordings()
                        # env.world.wait_for_tick()
                        env.world.tick()
                        Herdr.is_safe(env.controller_list)
                        if Herdr.safe == 1:
                            env.ped_space_count += 1
                        Herdr.GND_hist.append(Herdr.safe)
                        save_frame_count += 1
                        if save_frame_count % 100 == 0:
                            print(f"PING - I'm Alive - at {Herdr.vehicle.get_transform().location}")
                            print(f'Current Speed = {Herdr.vel_hist[-1]:.2f}')
                        if len(Herdr.collision_hist) > 0:
                            if Herdr.GND_hist[-1] == 0:
                                Herdr.GND_hist[-1] = 1
                            break

                    Herdr.done = True
                    # env.plot_hist_front.put('done')
                    print(f'--- DONE Run {i+1}/Round {round+1} at: {datetime.now().strftime("%d-%m-%Y--%H:%M:%S")} ---\n')
                    sim_time = save_frame_count/5
                    print(f'Sim Time est: {int(sim_time)} seconds')
                    # if int(sim_time) == 0:
                    #     Herdr.cleanup()
                    #     print(f'Real time: {time.time()-start_time:.4f}\n')
                    #     continue
                    total_sim_time += int(sim_time)
                    # job.join()
                    # job.close()
                    # start_step_time = time.time()
                    if recording_data:
                        env.new_run(Herdr.action_hist, Herdr.im_hist, Herdr.GND_hist, recording_data)
                        env.H5File.close()
                    ''' Calculate and Save SPL metric'''
                    # pl = env.pathlength(Herdr.pos_hist)
                    # env.SPL_hist.append([Herdr.success, pl, Herdr.p2pdist])
                    # spl = env.calc_SPL()
                    ''' Unused ploting tools '''
                    # env.top_writer.finish()
                    # env.writer.finish()
                    # plot_trajectory(np.asarray(env.pos_hist), torch.ones((len(env.pos_hist))), env.GOAL[0,0,:], collision=len(env.collision_hist))
                    # plt.savefig('./trajectory.png')
                    # plt.close('all')
                    ''' Empty memory and kill actors and sensors and save metrics'''
                    # writer.add_scalar("Validation/Run_time", sim_time, env.run)
                    # writer.add_scalar("Validation/In_Pedestrain_Space", env.ped_space_count/sim_time, env.run)
                    # writer.add_scalar("Validation/Distance_Traveled", EucDistance(Herdr.pos_hist[0],Herdr.pos_hist[-1]), env.run)
                    # writer.add_scalar("Validation/SPL", spl, env.run)
                    ''' Plot actions and optimal path at collision every 5th run '''
                    # if env.run % 5 == 0:
                    #     fig = plt.figure()
                    #     plot_args = [location2tensor(Herdr.vehicle.get_location()), Herdr.frame.permute(1,2,0), Herdr.event, Herdr.state, Herdr.planner.mean]
                    #     plot_action_cam_view(*plot_args)
                    #     fig.canvas.draw()
                    #     # Now we can save it to a numpy array.
                    #     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    #     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    #     data = data.transpose(2,0,1)/255
                    #     try:
                    #         writer.add_image("Validation/Context_Image", data, env.run)
                    #     finally:
                    #         del data, plot_args
                    #         plt.close(fig)
                    ''' Clean up from run '''
                    # writer.flush()
                    Herdr.cleanup()
                    print(f'Real time: {time.time()-start_time:.4f}\n')

            finally:
                ''' Train on collected data'''
                # run_data = carla_hdf5dataclass(f"/home/nathan/HERDR/all_carla_hdf5s/{recording_data}/", Herdr.horizon, imagefile_path=f"/home/nathan/HERDR/carla_images/{recording_data}/", counting=True, load_all_files=True)
                # test_sampler = SubsetRandomSampler(run_data.valid_start_indices)
                # testloader = torch.utils.data.DataLoader(run_data, sampler=test_sampler, batch_size=1)
                # run_data.set_pos_w(count_data_ratio(testloader))
                # testloader = torch.utils.data.DataLoader(run_data, sampler=test_sampler, batch_size=32)
                # opt = torch.optim.Adam(Herdr.model.parameters(), lr=1e-4, weight_decay=1e-2)
                # epoches = 2
                # for epoch in range(epoches):
                #     loss, pos_accuracy, accuracy, end_step = run_data.one_epoch(Herdr.model,testloader, start_step=end_step, writer=writer, opt=opt)
                #     print(f"{epoch+1} - Epoch Loss: {loss:.2f}, Epoch +Accuracy: {pos_accuracy:.2f}, Epoch Accuracy: {accuracy:.2f}, # steps: {len(testloader)}")
                #     writer.flush()
                # if loss < max_loss:
                #     max_loss = loss
                #     torch.save(Herdr.model, f'./models/carla{log_time}.pth')
                # del run_data, test_sampler, testloader, opt
                with open(f'/home/nathan/HERDR/pickles/training_counts_{log_time}.pkl','wb') as f:
                    pickle.dump([total_sim_time, env.run, end_step, env.SPL_hist],f)
                ''' Finish Run'''
                round += 1
                print(f'Total Sim Time in HRs:{int(total_sim_time/3600)}, in Mins:{int(total_sim_time/60)}')
                env.cleanup()
                # env.reworld()
                # writer.flush()
                env.reset()
                env.pop_map_pedestrians(num_peds=200)
    finally:
        Herdr.cleanup()
        env.world.tick()
        env.cleanup()
        # writer.close()
        del env, Herdr


def test():
    env = HERDRenv()
    env.reset()
    blk = 2
    Herdr = Herdragent(training=False, model_name='carla23-04-2022--09:34.pth',test_block=blk) # 'carla07-04-2022--14:41.pth'
    try:
        # env.reworld()
        # env.reset()
        env.pop_map_pedestrians(num_peds=200, test_block=blk)
        Herdr.reset()
        env.set_recordings()
        job = Process(target=env.background_save, args=(env.plot_hist_front,env.plot_hist_top,), daemon=True)
        job.start()
        start_time = time.time()
        save_frame_count = 0
        while not Herdr.done :
            Herdr.step()
            env.get_recordings(Herdr)
            env.world.tick()
            env.dist2peds(Herdr)
            Herdr.is_safe(env.controller_list)
            if Herdr.safe == 1:
                env.ped_space_count += 1
                print(f'In Pedestrain Space #{env.ped_space_count}')
            Herdr.GND_hist.append(Herdr.safe)
            save_frame_count += 1
            if save_frame_count % 100 == 0:
                print(f"PING - I'm Alive - at {Herdr.vehicle.get_transform().location}")
                print(f'Current Speed = {Herdr.vel_hist[-1]:.2f}')
            env.pos_hist.append(location2tensor(Herdr.vehicle.get_location()).numpy())
            if len(Herdr.collision_hist) > 0:
                if Herdr.GND_hist[-1] == 0:
                    Herdr.GND_hist[-1] = 1
                break
            if (time.time() - start_time >= 100):
                print(f'Timed-out')
                break
        Herdr.done = True
        env.plot_hist_front.put('done')
        job.join()
        job.close()
        pl = env.pathlength(Herdr.pos_hist)
        env.SPL_hist.append([Herdr.success, pl, Herdr.p2pdist])
        spl = env.calc_SPL()
        env.top_writer.finish()
        env.writer.finish()
        plot_trajectory(np.asarray(env.pos_hist), torch.ones((len(env.pos_hist))), Herdr.GOAL[0,0,:], collision=len(Herdr.collision_hist))
        plt.savefig('./trajectory.png')
        plt.close('all')
        Herdr.plot()
    
    finally:
        Herdr.cleanup()
        env.cleanup()
        env.world.tick()
        del env, Herdr    


def test_ped():
    env = HERDRenv()
    preset_list = dir(carla.WeatherParameters)[0:12]
    dict_WP = carla.WeatherParameters.__dict__
    env.world.set_weather(dict_WP[preset_list[random.randint(0,11)]])
    PED = Pedagent(test_block=2)
    try:
        # env.reworld()
        # env.reset()
        env.pop_map_pedestrians(num_peds=200)
        PED.reset()
        env.set_recordings()
        job = Process(target=env.background_save, args=(env.plot_hist_front,env.plot_hist_top,), daemon=True)
        job.start()
        start_time = time.time()
        save_frame_count = 0
        while not PED.done | (time.time() - start_time >= 300):
            PED.step()
            env.world.tick()
            PED.is_safe(env.controller_list)
            if PED.safe == 1:
                env.ped_space_count += 1
            PED.GND_hist.append(PED.safe)
            save_frame_count += 1
            if save_frame_count % 200 == 0:
                loc = carla.Location(x=float(PED.GOAL[0,0,0].numpy()), y=float(PED.GOAL[0,0,1].numpy()))
                PED.controller_walker.go_to_location(loc)
                print(f"PING - I'm Alive - at {PED.vehicle.get_transform().location}")
            env.pos_hist.append(location2tensor(PED.vehicle.get_location()).numpy())
            env.get_recordings(PED)
            if len(PED.collision_hist) > 0:
                if PED.GND_hist[-1] == 0:
                    PED.GND_hist[-1] = 1
                break
        PED.done = True
        env.plot_hist_front.put('done')
        job.join()
        job.close()
        pl = env.pathlength(PED.pos_hist)
        env.SPL_hist.append([PED.success, pl, PED.p2pdist])
        spl = env.calc_SPL()
        env.top_writer.finish()
        env.writer.finish()
        plot_trajectory(np.asarray(env.pos_hist), torch.ones((len(env.pos_hist))), PED.GOAL[0,0,:], collision=len(PED.collision_hist))
        plt.savefig('./trajectory.png')
        plt.close('all')
    
    finally:
        PED.cleanup()
        env.cleanup()
        print('Destroying Pedestrians')
        env.client.apply_batch([carla.command.DestroyActor(x.id) for x in env.actor_list])
        print('Destroying AI Controllers')
        [x.stop() for x in env.controller_list]
        env.client.apply_batch([carla.command.DestroyActor(x.id) for x in env.controller_list])
        print('Done.')
        env.world.tick()
        del env, PED  


def check_spawn_points():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    preset_list = dir(carla.WeatherParameters)[0:12]
    dict_WP = carla.WeatherParameters.__dict__
    world.set_weather(dict_WP[preset_list[random.randint(0,11)]])
    blueprint_library = world.get_blueprint_library()
    omafiets = blueprint_library.filter('omafiets')[0]
    transform = carla.Transform(carla.Location(x=82, y=-8, z=0.177637))
    vehicle = None
    start_time = time.time()
    actor_list = []
    # sensor_list = []
    try:
        while vehicle is None:
            vehicle = world.try_spawn_actor(omafiets, transform)
            time.sleep(0.3)
            if (start_time - time.time()) > 15:
                break
        actor_list.append(vehicle)
            
        time.sleep(15)
    finally:
        client.apply_batch([carla.command.DestroyActor(x.id) for x in actor_list])
        print('Destroying Agent')


if __name__ == '__main__':
    # main()
    test()
    # test_ped()
    # check_spawn_points()