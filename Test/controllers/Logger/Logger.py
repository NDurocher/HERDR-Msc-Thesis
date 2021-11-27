"""Logger controller."""

from controller import Supervisor
import numpy as np
import pickle


class Logger (Supervisor):
    def __init__(self):
        """Constructor: initialize constants."""
        Supervisor.__init__(self)
        self.cc = self.getSelf()
        self.timestep = int(self.getBasicTimeStep())
        self.hircus = self.getFromDef("Hircus")
        self.ped0 = self.getFromDef("Ped0")
        self.ped1 = self.getFromDef("Ped1")
        self.ped2 = self.getFromDef("Ped2")

    def fromhircus(self):
        pos = self.hircus.getPosition()
        if np.sqrt(pos[0] ** 2 + pos[2] ** 2) >= 9.5:
            self.reset()
        if self.getTime() > 50:
            self.reset()
        dist2goal = np.sqrt((pos[0] - GOAL[0, 0]) ** 2 + (pos[2] - GOAL[0, 1]) ** 2)
        if dist2goal < 0.5:
            # if within 1[m] of goal pause/end simulation
            print("Made it!!")
            self.reset()

    def reset(self):
        self.simulationSetMode(0)
        # self.simulationReset()
        self.ped1.restartController()
        self.ped2.restartController()
        self.hircus.restartController()
        pass

    def run(self):
        pedlist = [self.ped0, self.ped1, self.ped2]
        hircus_traj = []
        while (not self.step(self.timestep) == -1) or (not self.step(0) == -1):
            hircuspos = self.hircus.getPose()
            hircus_traj.append(hircuspos)
            dist_list = []
            for person in pedlist:
                pos = person.getPose()
                dist = np.sqrt((hircuspos[0]-pos[0])**2 + (hircuspos[2]-pos[2])**2)
                dist_list.append(dist)
            avg_dist = np.asarray(dist_list).mean()
            if self.simulationGetMode() == 0:
                self.simulationSetMode(1)
                break
        to_store = [hircus_traj, avg_dist]
        HERDRfile = open('mertics', 'ab')
        pickle.dump(to_store, HERDRfile)
        self.simulationReset()
        self.cc.restartController()


controller = Logger()
controller.run()

# Enter here exit cleanup code.
