from controller import Supervisor
import torch
from torchvision import transforms
from torchvision.utils import save_image
import sys
import numpy as np
import optparse
import math
from transforms3d.euler import mat2euler
from datetime import datetime

sys.path.insert(1, '/Users/NathanDurocher/Documents/GitHub/HERDR/src')
from Badgrnet import BadgrNet 
from actionplanner import BadgrPlan

WHEEL_RADIUS = 0.16 # m
WEBOTS_STEP_TIME = 100
DEVICE_SAMPLE_TIME = int(WEBOTS_STEP_TIME / 2) 
SCALE = 1000
GNSS_RATE = 1
HRZ=5
BATCH=150
GOAL = np.broadcast_to([-4.0, 2.0], (BATCH, 2)).copy()

class Hircus (Supervisor):
    """Control a Hircus PROTO."""

    def __init__(self, train=True):
        """Constructor: initialize constants."""
        Supervisor.__init__(self)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model.classes = [0]
        self.train = train
        self.ped1 = self.getFromDef("Ped1")
        self.ped2 = self.getFromDef("Ped2")
        self.hircus = self.getSelf()
        self.pose = self.hircus.getPose()
        self.frame = None

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
        
        # self.net = BadgrNet(Horizon=HRZ, Batch=BATCH)
        # self.net = torch.load('Herdr_1.pth', map_location=torch.device('cpu'))
        self.planner = BadgrPlan(Horizon=HRZ, vel_init=2.5)
        
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
    
    def enable_sensors(self, step_time):
        self.gps.enable(step_time)
        self.front_steer.getPositionSensor().enable(step_time)
        self.rear_steer.getPositionSensor().enable(step_time)
        self.body_imu.enable(step_time)
        self.gnss_heading_device.enable(step_time)
        self.Keyboard.enable(step_time)
        if not isinstance(self.camera, type(None)):
            self.camera.enable(step_time)
            self.camera.recognitionEnable(step_time)
            self.height = self.camera.getHeight()
            self.width = self.camera.getWidth()
            
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
        
        self.left_motor.setVelocity(speed)
        self.right_motor.setVelocity(speed)
        self.front_motor.setVelocity(speed)
        self.rear_motor.setVelocity(speed)

        self.front_steer.setPosition(steer)
        self.rear_steer.setPosition(-steer)
        self.front_steer.setVelocity(1)
        self.rear_steer.setVelocity(1)
        
    def recognize(self):
            self.recog = self.camera.getRecognitionObjects()
            if self.recog != []:
                obj = self.camera.getRecognitionObjects()
                self.obj = self.getFromId(obj[0].get_id())
                # print(self.obj)
                return 1
            else: 
                return 0   
    
    def reward(self):
        self.event = np.zeros(BATCH)
        self.pose = np.reshape(np.array(self.hircus.getPose()), [4, 4])
        state = self.calculate_position()
        # print(state)
        if self.train & self.recognize():
            ped_pos = np.array(self.obj.getPosition())
            self.obj.pose = np.reshape(np.array(self.obj.getPose()), [4, 4])
            ped_pos = np.broadcast_to(ped_pos, (HRZ, BATCH, 3)).copy()
            ped_ori = self.obj.pose[0:3, 0:3]
            self.event = self.is_safe(state, ped_pos, ped_ori)
            # event = np.sum(event, axis=0)
        elif (not self.train) & self.recognize():
            prediction = self.net(self.frame, self.actions)
            # event = torch.sum(prediction,0)
            print(self.event)
            self.event = event.detach().numpy()
        else:
            self.event = np.ones((HRZ,BATCH))
                
        self.event = self.event/np.linalg.norm(self.event, ord=1)                        
        goalReward = np.power( np.square((state[:,:,0]-GOAL[:,0])) + np.square((state[:,:,2]-GOAL[:,1])), 1)
        # goalReward = np.sum(goalReward, axis=0)
        # print(goalReward[0:6])
        goalReward = 1 - goalReward/np.linalg.norm(goalReward, ord=1)
        return 1.1*goalReward + self.event
            
    def calculate_position(self):
        euler = np.array(mat2euler(self.pose[0:3, 0:3]))
        new_pos = np.array(self.hircus.getPosition())
        new_state = np.hstack((new_pos.T, euler[1]))
        batch_state = np.broadcast_to(new_state, (BATCH,4)).copy()
        act = self.actions.numpy()
        state_stack = np.empty((HRZ,BATCH,4))
        # Y axis is vertical, movement is in X-Z plane 
        # [X Y Z Phi]
        for i in range(0, HRZ):
            batch_state[:,0] = batch_state[:,0] + (WEBOTS_STEP_TIME/SCALE)*np.cos(batch_state[:,3])*act[:,0,i]
            batch_state[:,2] = batch_state[:,2] + (WEBOTS_STEP_TIME/SCALE)*np.sin(batch_state[:,3])*act[:,0,i]
            batch_state[:,3] = batch_state[:,3] + (WEBOTS_STEP_TIME/SCALE)*act[:,1,i]*act[:,0,i]/self.wheelbase            
            state_stack[i,:,:] = batch_state.copy()
        return state_stack
        
    def is_safe(self, state, ped_pos, ped_ori):
        # Simple personal space model where [0,1]m is not safe, [1,2]m is less desirable and [2,inf] is safe
        kz = ped_ori[0,0]*2.25
        kx = ped_ori[0,2]*2.25
        check = np.sqrt( np.square((state[:,:,0]-ped_pos[:,:,0]-kx))/1.5**2 + np.square((state[:,:,2]-ped_pos[:,:,2]-kz))/2.5**2) - 1
        return check
    
    def reset(self):
        self.simulationReset()
        self.hircus.restartController()
        self.ped1.restartController()
        self.ped2.restartController()
        pass
        
    def Badgr(self):
        loader = transforms.Compose([transforms.ToTensor()])
        while not self.step(WEBOTS_STEP_TIME) == -1:
            frame = np.asarray(np.frombuffer(self.camera.getImage(), dtype=np.uint8))
            frame = np.reshape(np.ravel(frame), (self.height,self.width,4), order='C')
            frame = loader(frame[:,:,0:3]).float()
            self.frame = frame.unsqueeze(0)
            self.actions = self.planner.sample_new(batches=BATCH)
            # self.actions = self.actions.unsqueeze(0)
            
            r = self.reward()
            # r = r/np.linalg.norm(r, ord=1)
            best_r_arg = np.argmax(np.sum(r, axis=0))
            
            if self.recognize() & self.train:
                self.now = datetime.now()
                with open("Herdr_act.txt", "a") as f:
                    to_save = self.actions[best_r_arg,:,:].detach().numpy()
                    event_save = np.expand_dims(self.event[:,best_r_arg],0)
                    
                    # np.savetxt(f, to_save, '%2.5f', delimiter=',')
                    # f.write("%s.png\n" % str(self.now))
                    # np.savetxt(f, event_save, '%2.5f', delimiter=',')
                    # save_image(frame, str(self.now) +'.png')
                    
            
            self.update_motors(float(self.actions[best_r_arg,0,0]),float(self.actions[best_r_arg,1,0]))
            r = torch.tensor(r).transpose(0,1)
            self.planner.update_new(r, self.actions)
            pos = self.hircus.getPosition()
            if (abs(pos[0])  >= 9.5) | (abs(pos[2]) >= 9.5):
                self.reset()
            if self.getTime() > 50:
                self.reset()
            dist2goal = np.sqrt( (pos[0]-GOAL[0,0])**2 + (pos[2]-GOAL[0,1])**2 )
            if dist2goal < 1.5:
                # if within 1[m] of goal pause/end simulation
                self.reset()
    
        
controller = Hircus()
controller.Badgr()







    ####### Might not need
    # def recognizeYolo(self):
        # while not self.step(self.time_step) == -1:
            # img = np.asarray(np.frombuffer(self.camera.getImage(), dtype=np.uint8))
            # frame = np.reshape(np.ravel(img), (self.height,self.width,4), order='C')
            # results = self.model(frame)    

