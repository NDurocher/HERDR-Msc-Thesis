#!/usr/bin/env python

from cgi import test
from cgitb import handler
import glob
import os
import sys

from sklearn.preprocessing import SplineTransformer

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
from multiprocessing import Process, Queue
from Herdr_agent import Herdragent
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

from Badgrnet import HERDR
from actionplanner import HERDRPlan
from Carla_Trainer import carla_hdf5dataclass
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
    # ped_distance_hist = []
    ped_space_count = 0
    SPL_hist = []
    control_freq = 5 # Hz
    plot_hist_front = Queue()
    plot_hist_top = Queue()
    run = 0

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
        
    def pop_map_pedestrians(self, num_peds=200):
        # self.ped_actor_start_idx = len(self.actor_list)-1
        controller_bp = self.blueprint_library.find('controller.ai.walker')
        for i in range(num_peds):
            walker_bp = random.choice(self.blueprint_library.filter('walker.pedestrian.*'))
            trans = carla.Transform()
            trans.location = self.world.get_random_location_from_navigation()
            trans.location.z += 1

            walker_actor = self.world.try_spawn_actor(walker_bp, trans)
            # self.world.wait_for_tick()
            self.world.tick()
            if walker_actor is None:
                continue
            self.actor_list.append(walker_actor)

            controller_walker = self.world.try_spawn_actor(controller_bp, carla.Transform(), walker_actor)
            # self.world.wait_for_tick()
            self.world.tick()

            controller_walker.start()
            controller_walker.go_to_location(self.world.get_random_location_from_navigation())

            self.controller_list.append(controller_walker)
        # self.ped_actor_stop_idx = len(self.controller_list)-1
        del self.blueprint_library
        print('Pedestrians Added.')

    def reset(self):
        self.H5File = h5py.File(f'/home/nathan/HERDR/carla_hdf5s/{self.H5File_name}.h5', 'a')

    def pathlength(self, pos_list):
        np_pos_hist = np.asarray(pos_list)
        x, y = np_pos_hist[:,0], np_pos_hist[:,1]
        n = len(x)
        lv = [np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) for i in range(1, n)]
        L = sum(lv)
        return L
    
    def calc_SPL(self):
        sum_var = 0
        for success, length, p2pdist in self.SPL_hist:
            sum_var += success*(p2pdist/np.max([p2pdist,length,1e-9]))
        spl = 1/len(self.SPL_hist)*sum_var
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
        self.plot_hist_front.put([location2tensor(agent.vehicle.get_location()), agent.frame.permute(1,2,0), agent.event, agent.state])
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
    faulthandler.enable()
    try:
        log_time = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")
        # log_time = '31-03-2022--12:25:13'
        env = HERDRenv()
        # env.reworld()
        env.reset()
        env.pop_map_pedestrians(num_peds=200)
        writer = SummaryWriter(log_dir=f'/home/nathan/HERDR/carla_logs/{log_time}')
        Herdr = Herdragent(training=True, model_name=None)  # model_name=f'carla{log_time}.pth'
        if os.path.isfile(f'./training_counts_{log_time}.pkl'):
            with open(f'./training_counts_{log_time}.pkl','rb') as f:
                total_sim_time, env.run, end_step, env.SPL_hist = pickle.load(f)
        else:
            end_step = 0
            total_sim_time = 0
        round = 0

        # for j in range(10):
        while True:
            print(f"Round {round+1}!\n")
            H5File_name = f'{datetime.now()}'
            env.H5File_name = H5File_name
            # faulthandler.enable()
            try:
                for i in range(10):
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
                        if save_frame_count % 200 == 0:
                            print(f"PING - I'm Alive - at {Herdr.vehicle.get_transform().location}")
                        if len(Herdr.collision_hist) > 0:
                            if Herdr.GND_hist[-1] == 0:
                                Herdr.GND_hist[-1] = 1
                            break

                    Herdr.done = True
                    # env.plot_hist_front.put('done')
                    print(f'--- DONE Run {i+1}/Round {round+1} at: {datetime.now().strftime("%d-%m-%Y--%H:%M:%S")} ---\n')
                    sim_time = save_frame_count/5
                    print(f'Sim Time est: {int(sim_time)} seconds')
                    if int(sim_time) == 0:
                        Herdr.cleanup()
                        print(f'Real time: {time.time()-start_time:.4f}\n')
                        continue
                    total_sim_time += int(sim_time)
                    # job.join()
                    # job.close()
                    # start_step_time = time.time()
                    env.new_run(Herdr.action_hist, Herdr.im_hist, Herdr.GND_hist)
                    env.H5File.close()
                    pl = env.pathlength(Herdr.pos_hist)
                    env.SPL_hist.append([Herdr.success, pl, Herdr.p2pdist])
                    spl = env.calc_SPL()
                    # env.top_writer.finish()
                    # env.writer.finish()
                    # plot_trajectory(np.asarray(env.pos_hist), torch.ones((len(env.pos_hist))), env.GOAL[0,0,:], collision=len(env.collision_hist))
                    # plt.savefig('./trajectory.png')
                    # plt.close('all')
                    ''' Empty memory and kill actors and sensors and save metrics'''
                    Herdr.cleanup()
                    writer.add_scalar("Validation/Run_time", sim_time, env.run)
                    writer.add_scalar("Validation/In_Pedestrain_Space", env.ped_space_count/sim_time, env.run)
                    writer.add_scalar("Validation/Distance_Traveled", pl, env.run)
                    writer.add_scalar("Validation/SPL", spl, env.run)
                    if env.run % 50 == 0:
                        writer.add_image("Validation/Context_Image", Herdr.frame/255, 0)
                    print(f'Real time: {time.time()-start_time:.4f}\n')

            finally:
                ''' Train on collected data from current run '''
                run_data = carla_hdf5dataclass(f"/home/nathan/HERDR/carla_hdf5s/", Herdr.horizon, load_all_files=True)
                test_sampler = SubsetRandomSampler(run_data.valid_start_indices)
                testloader = torch.utils.data.DataLoader(run_data, sampler=test_sampler, batch_size=32)
                opt = torch.optim.Adam(Herdr.model.parameters(), lr=1e-4, weight_decay=1e-1)
                torch.set_default_dtype(torch.float32)
                loss, pos_accuracy, accuracy, end_step = run_data.one_epoch(Herdr.model,testloader, start_step=end_step, writer=writer, opt=opt)
                print(f"Epoch Loss: {loss:.2f}, Epoch +Accuracy: {pos_accuracy:.2f}, Epoch Accuracy: {accuracy:.2f}, # steps: {len(testloader)}")
                torch.save(Herdr.model, f'./models/carla{log_time}.pth')
                ''' Finish Run'''
                del run_data, test_sampler, testloader, opt
                # print("\nInsert Training Here\n")
                round += 1
                with open(f'./training_counts_{log_time}.pkl','wb') as f:
                    pickle.dump([total_sim_time, env.run, end_step, env.SPL_hist],f)
    finally:
        Herdr.cleanup()
        env.cleanup()
        print('Destroying Pedestrians')
        env.client.apply_batch([carla.command.DestroyActor(x.id) for x in env.actor_list])
        print('Destroying AI Controllers')
        [x.stop() for x in env.controller_list]
        env.client.apply_batch([carla.command.DestroyActor(x.id) for x in env.controller_list])
        print('Done.')
        del env, Herdr


def test():
    env = HERDRenv()
    Herdr = Herdragent(training=False, model_name='carla31-03-2022--12:25:13.pth')
    try:
        env.reset()
        # env.reworld()
        env.pop_map_pedestrians(num_peds=20)
        Herdr.reset()
        env.set_recordings()
        job = Process(target=env.background_save, args=(env.plot_hist_front,env.plot_hist_top,), daemon=True)
        job.start()
        start_time = time.time()
        save_frame_count = 0
        while not Herdr.done | (time.time() - start_time >= 600):
            Herdr.step()
            env.world.tick()
            Herdr.is_safe(env.controller_list)
            if Herdr.safe == 1:
                env.ped_space_count += 1
            Herdr.GND_hist.append(Herdr.safe)
            if len(Herdr.collision_hist) > 0:
                if Herdr.GND_hist[-1] == 0:
                    Herdr.GND_hist[-1] = 1
                break
            save_frame_count += 1
            if save_frame_count % 200 == 0:
                print(f"PING - I'm Alive - at {Herdr.vehicle.get_transform().location}")
            env.pos_hist.append(location2tensor(Herdr.vehicle.get_location()).numpy())
            env.get_recordings(Herdr)
        Herdr.done = True
        env.plot_hist_front.put('done')
        job.join()
        job.close()
        pl = env.pathlength(Herdr.pos_hist)
        env.SPL_hist.append([Herdr.success, pl, Herdr.p2pdist])
        env.top_writer.finish()
        env.writer.finish()
        plot_trajectory(np.asarray(env.pos_hist), torch.ones((len(env.pos_hist))), Herdr.GOAL[0,0,:], collision=len(Herdr.collision_hist))
        plt.savefig('./trajectory.png')
        plt.close('all')
    
    finally:
        Herdr.cleanup()
        env.cleanup()
        print('Destroying Pedestrians')
        env.client.apply_batch([carla.command.DestroyActor(x.id) for x in env.actor_list])
        print('Destroying AI Controllers')
        [x.stop() for x in env.controller_list]
        env.client.apply_batch([carla.command.DestroyActor(x.id) for x in env.controller_list])
        print('Done.')
        del env, Herdr    

if __name__ == '__main__':
    main()
    # test()