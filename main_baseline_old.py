"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 02 Jul 2017
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #

import os
import sys

from parameters import SAMPLE_AVG, TOTAL_RUNTIME, EPISODES, SHOULD_LOG, BATCH_SIZE, SAVE_MODEL, LOAD_MODEL, RENDER, wDelay, wResource, capRatio, DISCOUNT, succProb, MAP_ID, MODES, ARR_HORIZON, TOTAL_VESSEL, HORIZON, EVAL_EPISODES, ENV_VAR, OPENBLAS, OMP, MKL, NUMEXPR

if ENV_VAR:
    os.environ["OPENBLAS_NUM_THREADS"] = str(OPENBLAS)
    os.environ["OMP_NUM_THREADS"] = str(OMP)
    os.environ["MKL_NUM_THREADS"] = str(MKL)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUMEXPR)

from data import env_data
from environment_bsline import Maritime, countTable
from agents.tmin.tmin import tminAgent
from agents.pgFict_dcp.pg_fict_dcp import pg_fict_dcp
from utils import getTravelDelay, getVioCount
from scipy.stats import binom
import numpy as np
import networkx as nx
import platform
from pprint import pprint
import time
import auxLib as ax
from auxLib import average #, dumpDataStr
import math
import pdb
import rlcompleter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, help='Agent to run')
parser.add_argument('-r', '--runID', type=str, help='run ID')
args = parser.parse_args()

from numpy.random import multinomial

import csv
import math
# ============================================================================ #

print ("# ============================ START ============================ #")

# --------------------- Variables ------------------------------ #
ppath = os.getcwd() + "/"  # Project Path Location
#set random seed
SEED = int(args.runID)
np.random.seed(SEED )
ax.opFile = ""
# dirName = str(MAP_ID)+"_"+str(wDelay)+"_"+str(wResource)+"_"+str(capRatio)+"_"+str(HORIZON)+"_"+str(TOTAL_VESSEL)+"_"+args.agent

dirName = str(MAP_ID)+"_"+str(capRatio)+"_"+str(HORIZON)+"_"+str(TOTAL_VESSEL)+"_"+args.agent
FileName = str(MAP_ID)+"_"+str(capRatio)+"_"+str(HORIZON)+"_"+str(TOTAL_VESSEL)+"_"+args.agent+"_run"+args.runID

if LOAD_MODEL:
    ax.opFile = "./log/"+dirName+"/"+FileName +"_Inference"+".txt"
else:
    ax.opFile = "./log/" + dirName + "/" + FileName  + ".txt"

# --------------------- Main ---------------------------- #

def init():

    # ------------------------------ #
    # os.system("rm plots/reward.png")
    os.system("mkdir ./log/")
    os.system("mkdir ./log/" + dirName)

    # ax.deleteDir("./log/"+dirName+"/plots")
    ax.deleteDir("./runs")
    if not LOAD_MODEL:
        # ax.deleteDir("./log/"+dirName+"/model")
        os.system("mkdir ./log/"+dirName+"/model")

    os.system("rm " + ax.opFile)
    os.system("cp parameters.py ./log/"+dirName+"/")
    os.system("mkdir ./log/"+dirName+"/plots")
    os.system("mkdir ./log/" + dirName + "/model")
    # ax.createLog()

def initMap(cT, zGraph, dummyZONES):

    nDummy = len(dummyZONES)
    for dz in dummyZONES:
        nbr = nx.neighbors(zGraph, dz)
        if len(nbr) == 1:
            if TOTAL_VESSEL % nDummy > 0 and dz == 0:
                pop = int(TOTAL_VESSEL / nDummy) + int(TOTAL_VESSEL % nDummy)
            else:
                pop = int(TOTAL_VESSEL / nDummy)
            cT.nt_zz[dz][nbr[0]][0] = pop
            cT.nt_z[dz] = pop
        else:
            print ("Nbr of dummyZone > 1")
            exit()
    return cT

def mapCountTable(cT):

    cT_new = countTable()
    cT_new.nt_z = cT.nt_z
    for z in range(totalZONES):
        cT_new.nt_zt[z][1] = cT.nt_z[z]
    return cT_new

def getDiscountedReturn(tmpRew):

    tmpRew2 = []
    for t in range(len(tmpRew)):
        tmpRew2.append(tmpRew[t]*pow(DISCOUNT, t))
    return sum(tmpRew2)

def main():

    tStart = ax.now()
    runTime = 0
    init()
    #map specification
    data = env_data(mapName=MAP_ID)


    # print("planningZONES", data.planningZONES)
    # print("zGraph", data.zGraph)
    # print("totalResource", data.totalResource)
    # print("rCon", data.rCon)
    # print("rCap", data.rCap)
    # print("dummyNbrs", data.dummyNbrs)
    # print("totalZONES", data.totalZONES)
    # print("termZONES", data.termZONES)
    # print("dummyZONES", data.dummyZONES)
    # print("arrivalTime",data.arrivalTimeDict)
    # print("T_min_max", data.T_min_max)



    env = Maritime(data=data)
    agent = args.agent
    if agent == "pg_fict_dcp":
        mpa = pg_fict_dcp(load_model=LOAD_MODEL, dirName=dirName, FileName=FileName, data=data, seed = SEED)
    elif agent == "tmin":
        mpa = tminAgent(load_model=LOAD_MODEL, dirName=dirName, FileName=FileName, data=data, seed = SEED)
    else:
        ax.writeln("Error : Agent not defined!")
        exit()

    tEnd = ax.now()
    runTime += ax.getRuntime(tStart, tEnd)
    ax.writeln(" -------------------------------")
    ax.writeln(" > Compile Time : " + str(round(runTime, 3)) + " Seconds")
    ax.writeln(" -------------------------------")
    ax.writeln(" -------------------------------")
    ax.writeln(" > Training Starts...")
    ax.writeln(" -------------------------------")

    ax.writeln("------------------------------")
    ax.writeln("Agent : " + agent)
    ax.writeln("Map : "+str(MAP_ID))
    ax.writeln("HORIZON : " + str(HORIZON))
    ax.writeln("succProb : " + str(succProb))
    ax.writeln("Total Vessel : " + str(TOTAL_VESSEL))
    ax.writeln("Total Zones : " + str(data.totalZONES))
    ax.writeln("Planning Zones : " + str(data.planningZONES))
    ax.writeln("Start Zones : " + str(data.dummyZONES))
    ax.writeln("Terminal Zones : " + str(data.termZONES))
    ax.writeln("Total Resources : " + str(data.totalResource))
    ax.writeln("Delay Weight : " + str(wDelay))
    ax.writeln("Resource Weight : " + str(wResource))
    ax.writeln("Resource Capacity : " + str(data.rCap))
    ax.writeln("Resource Capacity : " + str(float(capRatio)))
    ax.writeln("Arrival Modes : " + str(MODES))
    ax.writeln("Arrival Horizon : " + str(ARR_HORIZON))
    ax.writeln("------------------------------")

    buffer_runAvg = np.zeros(SAMPLE_AVG)
    buffer_vioCountAvg = np.zeros(SAMPLE_AVG)
    buffer_trDelayAvg = np.zeros(SAMPLE_AVG)

    buffer_VioPenalty = np.zeros(SAMPLE_AVG)
    buffer_DelayPenalty = np.zeros(SAMPLE_AVG)



    sampleID = 0
    batchID = 0

    rollTime = 0
    sampling = 0
    actionTime = 0
    totalZONES = data.totalZONES

    vioCountHis = []

    for i_episode in range(1, EPISODES+1):

        #update target network 
        mpa.update_target()

        # ------- Init Environment
        cT = env.init()

        tStart2 = ax.now()
        if i_episode % SHOULD_LOG == 0:
            ax.writeln("------------------------------")
            ax.writeln("> Episode : "+ str(i_episode))
            beta_avg = {}
            for z in range(totalZONES):
                for zp in range(totalZONES):
                    beta_avg[(z, zp)] = []

        epRewardList = np.zeros(HORIZON+1)
        #performance metric
        epVioCountList = np.zeros((HORIZON+1, totalZONES))
        epVioCountHatList = np.zeros((HORIZON+1, totalZONES))
        epTrDelayList = np.zeros(HORIZON+1)
        #penalty at every time step
        epVioPen = np.zeros(HORIZON+1)
        epDelayPen = np.zeros(HORIZON+1)


        epCount = 0

        # --------- Buffers --------- #
        buffer_nt_z = np.zeros((HORIZON, totalZONES))
        buffer_ntz_zt = np.zeros((HORIZON, totalZONES, totalZONES, HORIZON+1))
        buffer_rt = np.zeros(HORIZON)
        buffer_rt_z = np.zeros((HORIZON, totalZONES))
        buffer_beta = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_nt_ztz = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_nt_z_new = np.zeros((HORIZON, totalZONES))
        t1_a = ax.now()
        mpa.ep_init()

        t1 = ax.now()
        for t in range(HORIZON):

            t1b = ax.now()
            beta = mpa.getBeta(cT, i_episode)
            learnedCost = mpa.getCost(cT, i_episode)[0]
            print(learnedCost)

            tmp2 = ax.getRuntime(t1b, ax.now())
            # print "action time", tmp2
            actionTime += tmp2

            buffer_nt_z[t] = cT.nt_z
            
            #array            
            vioCount = getVioCount(cT.nt_z, data.zGraph)
            vioCountHat = getVioCount(cT.nt_z * learnedCost, data.zGraph)

            t2 = ax.now()
            #global vioPen and delayPen
            Rt, RtHat, Rt_z, RtHat_z, cT_new , vioPen, vioPenHat, delayPen = env.step(t, cT, beta, learnedCost, mpa.myLambda_old)
            tmp1 = ax.getRuntime(t2, ax.now())
            sampling += tmp1
            # print "sampling ", tmp1

            buffer_ntz_zt[t] = cT_new.ntz_zt
            # buffer_rt_z[t] = RtHat_z
            buffer_rt_z[t] = Rt_z
            buffer_rt[t] = RtHat
            buffer_nt_ztz[t] = cT_new.nt_ztz
            buffer_beta[t] = beta[0]
            buffer_nt_z_new[t] = cT_new.nt_z


            # ------- Other Stats ------ #
            # vioCount = np.sum(getVioCount(cT_new.nt_z, data.zGraph))
            #can ignore trDelay so far
            trDelay = getTravelDelay(t, cT_new, data.zGraph)

            epRewardList[epCount] = Rt
            epVioCountList[epCount,:] = vioCount
            epVioCountHatList[epCount,:] = vioCountHat
            epTrDelayList[epCount] = trDelay
            epVioPen[epCount] = vioPen
            epDelayPen[epCount] = delayPen

            epCount += 1

            if i_episode % SHOULD_LOG == 0:
                # --------- Beta ---------- #
                betaStr = ""
                if t > 0 :
                    for z in range(totalZONES):
                        if (z not in data.dummyZONES) and (z not in data.termZONES):
                            for zp in nx.neighbors(data.zGraph, z):
                                # if cT.nt_z[z] > 0:
                                if buffer_nt_z[t][z] > 0:
                                    beta_avg[(z, zp)].append(round(beta[0][z][zp], 4))
                                betaStr += "b" + str(z) + str(zp) + " : " + str(round(beta[0][z][zp], 4)) + " "
                ax.writeln("  "+str(t)+" "+str(buffer_rt[t])+ " " +str(buffer_nt_z[t])+" | "+betaStr)
            cT = cT_new

        t1_b = ax.now()
        if i_episode % SHOULD_LOG == 0:
            ax.writeln("  " + str(t + 1) + " " + str("     ") + " " + str(cT_new.nt_z))


        # -------- Book Keeping ------- #
        # with learned cost
        t3 = ax.now()
        mpa.storeRollouts(buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt)
        # print "Indv Computation", ax.getRuntime(t3, ax.now())
        # print "total sampleing", sampling
        # print "action time", actionTime
        batchID += 1

        # ---------- Train Model --------- #
        if batchID % BATCH_SIZE == 0:
            t4 = ax.now()
            # mpa.trainLambda(i_episode, data.rCap)
            pg_loss_grad_norm = mpa.train(i_episode)
            mpa.trainLambda(i_episode, data.rCap)
            # print "Train", ax.getRuntime(t4, ax.now())
            mpa.clear()
            batchID = 0
            mpa.epoch += 1


        # epReward = getDiscountedReturn(epRewardList)
        # epVioPenDiscount = round(getDiscountedReturn(epVioPen), 3)
        # epDelayPenDiscount = round(getDiscountedReturn(epDelayPen), 3)

        # buffer_runAvg[sampleID] = epReward
        # buffer_VioPenalty[sampleID] = epVioPenDiscount
        # buffer_DelayPenalty[sampleID] = epDelayPenDiscount

        # buffer_vioCountAvg[sampleID] = np.sum(epVioCountList)
        # buffer_trDelayAvg[sampleID] = np.sum(epTrDelayList)

        # sampleID += 1







        ######
        #####
        #####meta gradient via cross validation
        #####
        cT = env.init()


        # epRewardList = np.zeros(HORIZON+1)
        # #performance metric
        # epVioCountList = np.zeros(HORIZON+1)
        # epTrDelayList = np.zeros(HORIZON+1)
        # #penalty at every time step
        # epVioPen = np.zeros(HORIZON+1)
        # epDelayPen = np.zeros(HORIZON+1)

        epCount = 0
        # --------- Buffers --------- #
        buffer_nt_z = np.zeros((HORIZON, totalZONES))
        buffer_ntz_zt = np.zeros((HORIZON, totalZONES, totalZONES, HORIZON+1))
        buffer_rt = np.zeros(HORIZON)
        buffer_rt_z = np.zeros((HORIZON, totalZONES))
        buffer_beta = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_nt_ztz = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_nt_z_new = np.zeros((HORIZON, totalZONES))

        mpa.ep_init()

        for t in range(HORIZON):

            beta = mpa.getBeta(cT, i_episode)
            learnedCost = mpa.getCost(cT, i_episode)[0]


        
            buffer_nt_z[t] = cT.nt_z

            #global vioPen and delayPen
            Rt, RtHat, Rt_z, RtHat_z, cT_new , vioPen, vioPenHat, delayPen = env.step(t, cT, beta, learnedCost, mpa.myLambda_old)
            tmp1 = ax.getRuntime(t2, ax.now())
            sampling += tmp1
            # print "sampling ", tmp1

            buffer_ntz_zt[t] = cT_new.ntz_zt
            buffer_rt_z[t] = RtHat_z
            buffer_rt[t] = RtHat
            buffer_nt_ztz[t] = cT_new.nt_ztz
            buffer_beta[t] = beta[0]
            buffer_nt_z_new[t] = cT_new.nt_z


            # # ------- Other Stats ------ #
            # vioCount = np.sum(getVioCount(cT_new.nt_z, data.zGraph))
            # #can ignore trDelay so far
            # trDelay = getTravelDelay(t, cT_new, data.zGraph)

            # epRewardList[epCount] = Rt
            # epVioCountList[epCount] = vioCount
            # epTrDelayList[epCount] = trDelay
            # epVioPen[epCount] = vioPen
            # epDelayPen[epCount] = delayPen

            epCount += 1
            cT = cT_new




        # -------- Book Keeping ------- #
        t3 = ax.now()
        #compute value function
        mpa.storeRollouts(buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt)
        #
        # mpa.storeRolloutsCost(buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt)
 
        batchID += 1

        # ---------- Train Model --------- #
        if batchID % BATCH_SIZE == 0:
            loss_meta, loss_1, loss_2, c1, c2 = mpa.trainMeta(i_episode, buffer_nt_z, buffer_ntz_zt)
            #update lambda
            # print "Train", ax.getRuntime(t4, ax.now())
            mpa.clear()
            batchID = 0


        # --------- Logs ------- #
        if i_episode % SHOULD_LOG == 0:
            
            #epVioCountList, epVioCountHatList are array
            # epVioCount = np.sum(epVioCountList)
            # epVioCountHat = np.sum(epVioCountHatList)
            epTrDelay = np.sum(epTrDelayList)
            mpa.logStat(i_episode, epVioCountList, epVioCountHatList, epTrDelay, mpa.myLambda, mpa.myXi, loss_meta, loss_1, loss_2, c1, c2, pg_loss_grad_norm)
            # os.system("cp -r runs/* ./log/"+dirName+"/plots/"+FileName)

        # # --------- testing ------- #
        # #there is no exploration for beta
        # if i_episode % 10 == 0:
        #     testEpReward = []
        #     testEpDelay = []
        #     testEpVio = []

        #     for i_test in range(10):
        #         # ------- Init Environment
        #         cT = env.init()
        #         epRewardList = np.zeros(HORIZON+1)
        #         #performance metric
        #         epVioCountList = np.zeros(HORIZON+1)
        #         epTrDelayList = np.zeros(HORIZON+1)
        #         #penalty at every time step
        #         epVioPen = np.zeros(HORIZON+1)
        #         epDelayPen = np.zeros(HORIZON+1)

        #         epCount = 0
        #         mpa.ep_init()

        #         for t in range(HORIZON):
        #             beta = mpa.getBeta(cT, i_test)
        #             #global vioPen and delayPen
        #             Rt, Rt_z, cT_new , vioPen, delayPen = env.step(t, cT, beta, mpa.myLambda)
        #             # ------- Other Stats ------ #
        #             vioCount = np.sum(getVioCount(cT_new.nt_z, data.zGraph))
        #             #can ignore trDelay so far
        #             trDelay = getTravelDelay(t, cT_new, data.zGraph)

        #             epRewardList[epCount] = Rt
        #             epVioCountList[epCount] = vioCount
        #             epTrDelayList[epCount] = trDelay
        #             epVioPen[epCount] = vioPen
        #             epDelayPen[epCount] = delayPen

        #             epCount += 1
        #             cT = cT_new
        #         testEpReward.append(getDiscountedReturn(epRewardList))
        #         testEpDelay.append(np.sum(epVioCountList)) 
        #         testEpVio.append(np.sum(epTrDelayList))


        #     mpa.logTest(int(i_episode / 10), np.mean(testEpReward), np.mean(testEpVio), np.mean(testEpDelay))
        #     os.system("cp -r runs/* ./log/"+dirName+"/plots/")


    ax.writeln("\n--------------------------------------------------")
    totalRunTime = ax.getRuntime(tStart, ax.now())
    ax.writeln("Total Runtime : " + str(round(totalRunTime, 3)) + " Seconds")
    ax.writeln("--------------------------------------------------")
    mpa.writer.close()

    print(vioCountHis)



# =============================================================================== #

if __name__ == '__main__':



    main()
    print ("\n\n# ============================  END  ============================ #")
    """
    sbm = ax.seaBplotsMulti_Bar(HORIZON)
    zone_count = updateZoneCount(t, nt[t])
    zCount.append(list(zone_count))
    plot(sbm, t, zone_count)
    sbm.save("plots/count_large")
    ax.joinPNG(ppath+"plots/")
    """

