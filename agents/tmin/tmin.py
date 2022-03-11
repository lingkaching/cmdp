"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 16 Jul 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys
import os
from pprint import pprint
import time
import auxLib as ax
import pdb
import rlcompleter
import numpy as np
import networkx as nx
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
from tensorboardX import SummaryWriter
# =============================== Variables ================================== #

# ============================================================================ #
class tminAgent:

    def __init__(self, data=None, load_model=False, dirName=""):

        self.totalZONES = data.totalZONES
        self.dummyZONES = data.dummyZONES
        self.termZONES = data.termZONES
        self.zGraph = data.zGraph
        self.epoch = 0
        self.writer = SummaryWriter()
        return
    def getBeta(self, cT, i_episode):

        return np.zeros((1, self.totalZONES, self.totalZONES))

    def train(self, i_episode):

        return

    def storeRollouts(self, buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt):

        return

    def clear(self):
        return

    def logBK(self, i_episode, epReward, betaAvg2, sampleAvg):

        self.writer.add_scalar('Total Rewards', epReward, i_episode)
        self.writer.add_scalar('Sample Avg. Rewards', sampleAvg, i_episode)
        for z in range(self.totalZONES):
            if (z not in self.dummyZONES) and (z not in self.termZONES):
                for zp in nx.neighbors(self.zGraph, z):
                    # pdb.set_trace()
                    self.writer.add_scalar("Beta/"+str(z)+"_"+str(zp), betaAvg2[z][zp], i_episode)
    
    def log(self, i_episode, epReward, betaAvg2, vio, delay):

            self.writer.add_scalar('Total Rewards', epReward, i_episode)
            self.writer.add_scalar('Total ResVio', vio, i_episode)
            self.writer.add_scalar('Total Delay', delay, i_episode)

            # self.writer.add_scalar('Sample Avg. Rewards', sampleAvg, i_episode)
            # self.writer.add_scalar('Loss', self.loss, i_episode)
            for z in range(self.totalZONES):
                if (z not in self.dummyZONES) and (z not in self.termZONES):
                    for zp in nx.neighbors(self.zGraph, z):
                        # pdb.set_trace()
                        self.writer.add_scalar("Beta/"+str(z)+"_"+str(zp), betaAvg2[z][zp], i_episode)
    
    def save_model(self):

        return

    def ep_init(self):
        return

# =============================================================================== #

if __name__ == '__main__':
    main()
    