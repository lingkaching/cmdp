"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 02 Jul 2017
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# ================================ priImports ================================ #

import sys
import os
import platform
from pprint import pprint
import time
import auxLib as ax
from auxLib import average

# ================================ secImports ================================ #
import numpy as np
import networkx as nx
from utils import cost_t_nt_z, getMaxReward
import pdb
import rlcompleter
import random
from numpy.random import multinomial
# ============================================================================ #

# --------------------- Variables ------------------------------ #

ppath = os.getcwd() + "/"  # Project Path Location

# -------------------------------------------------------------- #

class countTable:

    def __init__(self, totalZONES, HORIZON, NUM_ACTIONS):
        self.nt_zt = np.zeros((totalZONES, HORIZON + 1))   # n_t(z, \tau = t)
        self.nt_zz = np.zeros((totalZONES, totalZONES, HORIZON + 1)) # n_t(z, z', \tau > t)
        self.nt_zt_zt = np.zeros((totalZONES, HORIZON + 1, totalZONES, HORIZON + 1)) # n_t(z, \tau, z', \tau')
        self.nt_z = np.zeros(totalZONES)   # n_t(z)
        self.ntz_zt = np.zeros((totalZONES, totalZONES, HORIZON+1))
        # self.nt_ztz = np.zeros((totalZONES, totalZONES)) # n_t(z,z')

        self.ns = np.zeros((totalZONES, totalZONES))
        self.nsa = np.zeros((totalZONES, totalZONES, NUM_ACTIONS))



class Maritime:

    def __init__(self, data=None):

        np.random.seed(data.seed)
        random.seed(data.seed)

        self.map_id = data.map_id
        self.totalZONES = data.totalZONES
        self.dummyZONES = data.dummyZONES
        self.termZONES = data.termZONES
        self.Pzz_mat = data.Pzz_mat
        self.planningZONES = data.planningZONES
        self.T_min_max = data.T_min_max

        self.batch_size = data.batch_size
        self.zGraph = data.zGraph
        self.num_options = data.num_options
        self.num_actions = data.num_actions

        self.renderFlag = False
        self.zCount = None
        self.zCountList = [[] for _ in range(self.batch_size)]
        self.zCountList_1hot = [[] for _ in range(self.batch_size)]
        self.cT = countTable
        self.t = None
        self.reward = []
        self.rAvg = 0
        self.rAvgList = []
        self.beta_all = np.zeros((data.HORIZON+2, data.totalZONES, data.totalZONES))
        self.maxReward = getMaxReward()
        if "real" not in self.map_id:
            self.arr_seeds = data.arr_seeds
            self.total_vessels = data.total_vessels
            self.arrivalTimeDict = data.arrivalTimeDict
            self.HORIZON = data.HORIZON
        else:
            self.arrival_dist = data.arrival_dist
            self.initialCount = data.initialCount
            self.HORIZON = data.real_horizon


        # ----- Discrete
        self.action_space = data.action_space
        self.num_actions = data.num_actions

    def getFailVessel(self, tmpSuccVessel, total_horizon):
        totalHorizon = total_horizon
        tmpSuccVessel.append(0)
        tmpPop = sum(tmpSuccVessel)
        tmpFailVessel = [0]
        for tt in range(1, totalHorizon + 1):
            tmpFail = tmpPop - sum(tmpSuccVessel[1:tt + 1])
            tmpFailVessel.append(tmpFail)
        return tmpSuccVessel, tmpFailVessel

    def init(self):

        cT = countTable(self.totalZONES, self.HORIZON, self.num_actions)
        if "real" in self.map_id:
            arr = []
            arr_count = 0
            i = 0
            for dz in self.dummyZONES:
                arr.append(self.arrival_dist[dz].rvs()[0].tolist())
                arr_count += sum(arr[i])
                i += 1
            tmpArrivalTime = {}
            for dz in self.dummyZONES:
                dzID = self.dummyZONES.index(dz)
                tmpArrivalTime[dz] = {}
                tmpSuccVessel, tmpFailVessel = self.getFailVessel(arr[dzID], self.HORIZON)
                count = -1
                tmpArrivalTime[dz] = {}
                tmpArrivalTime[dz]['succ'] = {}
                tmpArrivalTime[dz]['fail'] = {}
                for tt in range(self.HORIZON):
                    count += 1
                    if tmpSuccVessel[count] != 0:
                        tmpArrivalTime[dz]['succ'][tt] = tmpSuccVessel[count]
                        tmpArrivalTime[dz]['fail'][tt] = tmpFailVessel[count]
            self.arrivalTime = tmpArrivalTime
            iniCount = self.initialCount.rvs()[0]
            cT.nt_z = iniCount.copy()
            cT.nt_zt[:,0] = iniCount.copy()
            for dz in self.dummyZONES:
                dzID = self.dummyZONES.index(dz)
                nbr = nx.neighbors(self.zGraph, dz)
                if len(nbr) == 1:
                    cT.nt_zz[dz][nbr[0]][0] = sum(arr[dzID])
                    cT.nt_z[dz] = sum(arr[dzID])
                else:
                    print ("Nbr of dummyZone > 1")
                    exit()
            self.total_vessels = int(iniCount.copy().sum() + arr_count)
        else:
            # -------- Arrival Distribution
            id = random.choice(self.arr_seeds)
            self.arrivalTime = self.arrivalTimeDict[id]

            # -------- Initial Count
            nDummy = len(self.dummyZONES)
            for dz in self.dummyZONES:
                nbr = nx.neighbors(self.zGraph, dz)
                if len(nbr) == 1:
                    if self.total_vessels % nDummy > 0 and dz == 0:
                        pop = int(self.total_vessels / nDummy) + int(self.total_vessels % nDummy)
                    else:
                        pop = int(self.total_vessels / nDummy)
                    cT.nt_zz[dz][nbr[0]][0] = pop
                    cT.nt_z[dz] = pop
                else:
                    print ("Nbr of dummyZone > 1")
                    exit()
        return cT

    def step(self, t, cT, action_prob):

        self.action_prob = action_prob
        # self.beta_all[t] = beta
        self.cT = cT
        self.t = t
        '''Required only during rendering'''
        # self.zCount = updateZoneCount(self.t, self.cT, self.instance.zGraph)
        # self.reward.append(self.rt)  # Required for rendering

        # ---------- Rewards ----------- #
        self.rt_z, res, delay = self._atomicReward(self.cT.nt_z)
        self.rt = np.sum(self.rt_z * self.cT.nt_z)
        res = np.sum(res * self.cT.nt_z)
        delay = np.sum(delay * self.cT.nt_z)

        # Normalize the reward
        self.rt_z = self.rt_z / self.maxReward

        # -------------------------------------------- #
        self.cT = self._sample(t, self.cT)
        return self.rt, self.rt_z, self.cT, res, delay, self.beta_all

    def _atomicReward(self, nt_z):

        rm = np.zeros((self.totalZONES))
        res = np.zeros((self.totalZONES))
        delay = np.zeros((self.totalZONES))
        for z in self.planningZONES:
            # tmp[z] = -1 * cost_t_nt_z(z, nt_z[z])
            rm_z, res_z, delay_z = cost_t_nt_z(z, nt_z[z])
            rm[z] = -1 * rm_z
            res[z] = -1 * res_z
            delay[z] = -1 * delay_z
        return rm, res, delay

    def _sample(self, t, cT):

        cT.ns.fill(0)
        cT.nsa.fill(0)
        self.beta_all = np.zeros((1, self.totalZONES, self.totalZONES))
        zGraph = self.zGraph
        # ------- Sample z' --------- #
        # sample n_t(z, \tau=t, z') from n_t(z, \tau=t)
        ztz = np.zeros((self.totalZONES, self.totalZONES))
        for z in self.planningZONES:
            pBucket = self.Pzz_mat[z]
            if sum(pBucket) > 0:
                ztz[z] = np.random.multinomial(cT.nt_zt[z][t], pBucket)

        # -------- update nt_ztz
        cT.nt_ztz = ztz
        cT.ns = ztz

        # ------- Sample \tau' ------ #
        # --- New
        # sample nt(z, t, z', t+k ) from nt(z, t, z')
        notReached = np.zeros(self.totalZONES)
        for z in self.planningZONES:
            for zp in nx.neighbors(zGraph, z):      

                avg_beta = round(np.multiply(self.action_prob[z][zp], self.action_space).sum(), 2)
                self.beta_all[0][z][zp] = avg_beta

                # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
                # pdb.set_trace()

                tMin = self.T_min_max[z][zp][0]
                tMax = self.T_min_max[z][zp][1]
                n = tMax - tMin
                n_zz = int(cT.ns[z][zp])
                cT.nsa[z][zp] = multinomial(n_zz, self.action_prob[z][zp])
                # countBuckets = []
                countBuckets = np.zeros(n+1)
                for b_id in range(self.num_actions):
                    beta = self.action_space[b_id]
                    if cT.nsa[z][zp][b_id] > 0:
                        m = cT.nsa[z][zp][b_id]
                        tmpSample = np.random.binomial(n, beta, int(m))                
                        for c in range(n + 1):
                            # countBuckets.append(np.count_nonzero(tmpSample == c))
                            countBuckets[c] += np.count_nonzero(tmpSample == c)
                if len(cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1]) < len(countBuckets):
                    len_nt = len(cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1])
                    len_pB = len(countBuckets)
                    cT.nt_zt_zt[z][t][zp][t + tMin:t + tMax + 1] = countBuckets[0:len_nt]
                    if sum(countBuckets[len_nt:len_pB]) > 0:
                        notReached[z] += sum(countBuckets[len_nt:len_pB])
                else:
                    cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1] = countBuckets


        # -------- Update Count Tables for next system time ------- #
        for z in range(self.totalZONES):
            if z not in self.dummyZONES:
                # ------- cT.nt_zt (Newly Arrived Agents) ------- #
                cT.nt_zt[z][t+1] = sum(cT.nt_zz[:,z,t+1]) + sum(cT.nt_zt_zt[:,t,z,t+1]) + notReached[z]

                # ------- cT.nt_zz (Transiting Agents)------- #
                for zp in range(self.totalZONES):
                    cT.nt_zz[z][zp][t+2:] = cT.nt_zz[z][zp][t+2:] + cT.nt_zt_zt[z][t][zp][t+2:]

                # ------ cT.nt_z ( Total Agents present in zone z)------ #
                cT.nt_z[z] = cT.nt_zt[z][t+1] + sum(map(lambda zpp : sum(cT.nt_zz[z][zpp][t+2:]), [i for i in range(self.totalZONES)]))
                # ------ cT.ntz_zt (used in fictitious)----- #
                cT.ntz_zt[z] = cT.nt_zt_zt[z][t]


        # ------- Update Dummy Zone Count ------ #
        for dz in self.dummyZONES:
            for tp in self.arrivalTime[dz]['succ']:
                if t + 1 == tp:
                    succVessel = self.arrivalTime[dz]['succ'][tp]
                    failVessel = self.arrivalTime[dz]['fail'][tp]
                    pz = nx.neighbors(zGraph, dz)[0]
                    cT.nt_zt[pz][t+1] += succVessel
                    cT.nt_z[pz] += succVessel
                    cT.nt_z[dz] = failVessel


        # ------- Update Terminal Zone Count -------- #
        for z in self.termZONES:
            cT.nt_zt[z][t + 1] += cT.nt_zt[z][t]
            cT.nt_z[z] = cT.nt_zt[z][t+1] + sum(map(lambda zpp : sum(cT.nt_zz[z][zpp][t+2:]), [i for i in range(self.totalZONES)]))

        assert sum(cT.nt_z) == self.total_vessels, "Count Error : "+str(cT.nt_z)
        return  cT

    def _sample_BK(self, t, cT):

        zGraph = self.zGraph
        # ------- Sample z' --------- #
        # sample n_t(z, \tau=t, z') from n_t(z, \tau=t)
        ztz = np.zeros((self.totalZONES, self.totalZONES))
        for z in self.planningZONES:
            pBucket = self.Pzz_mat[z]
            if sum(pBucket) > 0:
                ztz[z] = np.random.multinomial(cT.nt_zt[z][t], pBucket)

        # -------- update nt_ztz
        cT.nt_ztz = ztz

        # ------- Sample \tau' ------ #
        # --- New
        # sample nt(z, t, z', t+k ) from nt(z, t, z')
        notReached = np.zeros(self.totalZONES)
        for z in self.planningZONES:
            for zp in nx.neighbors(zGraph, z):

                

                tMin = self.T_min_max[z][zp][0]
                tMax = self.T_min_max[z][zp][1]
                n = tMax - tMin


                beta = self.beta_all[t][z][zp]
                
                
                
                
                # print t, z, zp, beta
                tmpSample = np.random.binomial(n, beta, int(ztz[z][zp]))
                countBuckets = []
                for c in range(n + 1):
                    countBuckets.append(np.count_nonzero(tmpSample == c))





                if sum(countBuckets) > 0:
                    pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
                    pdb.set_trace()





                if len(cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1]) < len(countBuckets):
                    len_nt = len(cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1])
                    len_pB = len(countBuckets)
                    cT.nt_zt_zt[z][t][zp][t + tMin:t + tMax + 1] = countBuckets[0:len_nt]
                    if sum(countBuckets[len_nt:len_pB]) > 0:
                        notReached[z] += sum(countBuckets[len_nt:len_pB])

                else:
                    cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1] = countBuckets


        # -------- Update Count Tables for next system time ------- #
        for z in range(self.totalZONES):
            if z not in self.dummyZONES:
                # ------- cT.nt_zt (Newly Arrived Agents) ------- #
                cT.nt_zt[z][t+1] = sum(cT.nt_zz[:,z,t+1]) + sum(cT.nt_zt_zt[:,t,z,t+1]) + notReached[z]

                # ------- cT.nt_zz (Transiting Agents)------- #
                for zp in range(self.totalZONES):
                    cT.nt_zz[z][zp][t+2:] = cT.nt_zz[z][zp][t+2:] + cT.nt_zt_zt[z][t][zp][t+2:]

                # ------ cT.nt_z ( Total Agents present in zone z)------ #
                cT.nt_z[z] = cT.nt_zt[z][t+1] + sum(map(lambda zpp : sum(cT.nt_zz[z][zpp][t+2:]), [i for i in range(self.totalZONES)]))
                # ------ cT.ntz_zt (used in fictitious)----- #
                cT.ntz_zt[z] = cT.nt_zt_zt[z][t]


        # ------- Update Dummy Zone Count ------ #
        for dz in self.dummyZONES:
            for tp in self.arrivalTime[dz]['succ']:
                if t + 1 == tp:
                    succVessel = self.arrivalTime[dz]['succ'][tp]
                    failVessel = self.arrivalTime[dz]['fail'][tp]
                    pz = nx.neighbors(zGraph, dz)[0]
                    cT.nt_zt[pz][t+1] += succVessel
                    cT.nt_z[pz] += succVessel
                    cT.nt_z[dz] = failVessel


        # ------- Update Terminal Zone Count -------- #
        for z in self.termZONES:
            cT.nt_zt[z][t + 1] += cT.nt_zt[z][t]
            cT.nt_z[z] = cT.nt_zt[z][t+1] + sum(map(lambda zpp : sum(cT.nt_zz[z][zpp][t+2:]), [i for i in range(self.totalZONES)]))

        assert sum(cT.nt_z) == self.total_vessels, "Count Error : "+str(cT.nt_z)
        return  cT
