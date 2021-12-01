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




controller = Logger()
controller.run()

# Enter here exit cleanup code.
