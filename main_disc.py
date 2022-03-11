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

from parameters import SAMPLE_AVG, TOTAL_RUNTIME, SEED, EPISODES, SHOULD_LOG, BATCH_SIZE, SAVE_MODEL, LOAD_MODEL, RENDER, wDelay, wResource, capRatio, DISCOUNT, succProb, MAP_ID, MODES, ARR_HORIZON, TOTAL_VESSEL, HORIZON, EVAL_EPISODES, ENV_VAR, OPENBLAS, OMP, MKL, NUMEXPR

if ENV_VAR:
    os.environ["OPENBLAS_NUM_THREADS"] = str(OPENBLAS)
    os.environ["OMP_NUM_THREADS"] = str(OMP)
    os.environ["MKL_NUM_THREADS"] = str(MKL)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUMEXPR)

from data import env_data
from environment_disc import Maritime, countTable
from agents.random.randomAgent import randomAgent
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
args = parser.parse_args()

from numpy.random import multinomial

import csv
import math
# ============================================================================ #

print ("# ============================ START ============================ #")

# --------------------- Variables ------------------------------ #
ppath = os.getcwd() + "/"  # Project Path Location
np.random.seed(SEED)
ax.opFile = ""
dirName = str(MAP_ID)+"_"+str(wDelay)+"_"+str(wResource)+"_"+str(capRatio)+"_"+str(HORIZON)+"_"+str(TOTAL_VESSEL)+"_"+args.agent

if LOAD_MODEL:
    ax.opFile = "./log/"+dirName+"/"+dirName+"_Inference"+".txt"
else:
    ax.opFile = "./log/" + dirName + "/" + dirName + ".txt"

# --------------------- Main ---------------------------- #

def init():

    # ------------------------------ #
    # os.system("rm plots/reward.png")
    os.system("mkdir ./log/")
    os.system("mkdir ./log/" + dirName)

    ax.deleteDir("./log/"+dirName+"/plots")
    ax.deleteDir("./runs")
    if not LOAD_MODEL:
        ax.deleteDir("./log/"+dirName+"/model")
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

def evaluate(data, env, agent):

    ax.opFile = "./log/"+dirName+"/"+dirName+"_eval"+".txt"
    os.system("rm " + ax.opFile)

    tStart = ax.now()
    runTime = 0
    buffer_Return = np.zeros(EVAL_EPISODES)
    buffer_Vio = np.zeros(EVAL_EPISODES)
    buffer_Delay = np.zeros(EVAL_EPISODES)
    buffer_VioPenalty = np.zeros(EVAL_EPISODES)
    buffer_DelayPenalty = np.zeros(EVAL_EPISODES)

    totalZONES = data.totalZONES
    ax.writeln("----------------------------------------- EVALUATION STARTS ------------------------------------------")

    for i_episode in range(1, EVAL_EPISODES+1):

        # ------- Init Environment
        cT = env.init()


        tStart2 = ax.now()
        if i_episode % SHOULD_LOG == 0:
            ax.writeln("------------------------------")
            ax.writeln("> Eval Episode : "+ str(i_episode))
            beta_avg = {}
            for z in range(totalZONES):
                for zp in range(totalZONES):
                    beta_avg[(z, zp)] = []


        epRewardList = np.zeros(HORIZON+1)
        epVioCountList = np.zeros(HORIZON+1)
        epTrDelayList = np.zeros(HORIZON+1)
        epVioPen = np.zeros(HORIZON+1)
        epDelayPen = np.zeros(HORIZON+1)

        epCount = 0
        # --------- Buffers --------- #
        buffer_nt_z = np.zeros((HORIZON, totalZONES))
        for t in range(HORIZON):
            beta = agent.getBeta(cT, i_episode)
            buffer_nt_z[t] = cT.nt_z
            Rt, _, cT_new,  vioPen, delayPen  = env.step(t, cT, beta)

            # ------- Other Stats ------ #
            vioCount = np.sum(getVioCount(cT_new.nt_z, data.zGraph))
            trDelay = getTravelDelay(t, cT_new, data.zGraph)

            epRewardList[epCount] = Rt
            epVioCountList[epCount] = vioCount
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
                                if cT.nt_z[z] > 0:
                                    beta_avg[(z, zp)].append(round(beta[0][z][zp], 4))
                                betaStr += "b" + str(z) + str(zp) + " : " + str(round(beta[0][z][zp], 4)) + " "
                ax.writeln("  "+str(t)+" "+str(Rt)+ " " +str(buffer_nt_z[t])+" | "+betaStr)
            cT = cT_new

        if i_episode % SHOULD_LOG == 0:
            ax.writeln("  " + str(t + 1) + " " + str("     ") + " " + str(cT_new.nt_z))



        epReward = getDiscountedReturn(epRewardList)
        epVioPenDiscount = round(getDiscountedReturn(epVioPen), 3)
        epDelayPenDiscount = round(getDiscountedReturn(epDelayPen), 3)


        buffer_Return[i_episode-1] = epReward
        buffer_VioPenalty[i_episode-1] = epVioPenDiscount
        buffer_DelayPenalty[i_episode-1] = epDelayPenDiscount

        buffer_Vio[i_episode-1] = np.sum(epVioCountList)
        buffer_Delay[i_episode-1] = np.sum(epTrDelayList)

        if runTime > TOTAL_RUNTIME:
            break
        # --------- Logs ------- #
        if i_episode % SHOULD_LOG == 0:
            epVioCount = np.sum(epVioCountList) # average(epVioCountList)
            epTrDelay = np.sum(epTrDelayList)
            # epResVio = np.sum(epResVioList)
            # epResVioNoPen = float(np.sum(epResVioNoPenList))/(len(planningZONES) * (HORIZON-1))

            ax.writeln("\n\n  Total Episode Reward : "+ str(round(epReward, 2)))
            ax.writeln("  Total Episode VioPenalty : " + str(round(epVioPenDiscount, 2)))
            ax.writeln("  Total Episode DelayPenalty : " + str(round(epDelayPenDiscount, 2)))

            ax.writeln("  ")
            ax.writeln("  Total Res. Vio : " + str(round(np.sum(epVioCountList), 2)))
            ax.writeln("  Total Travel Delay : " + str(round(np.sum(epTrDelayList), 2)))
            ax.writeln("  ")
            ax.writeln("  Avg. Res. Vio : " + str(round(np.average(epVioCountList), 2))+" std : "+str(round(np.std(epVioCountList), 2)))
            ax.writeln("  Avg. Travel Delay : " + str(round(np.average(epTrDelayList), 2))+" std : "+str(round(np.std(epTrDelayList), 2)))
            ax.writeln("  ")
            ax.writeln("  Max Res. Vio : " + str(round(np.max(epVioCountList), 2)))
            ax.writeln("  Max Travel Delay : " + str(round(np.max(epTrDelayList), 2)))

            # -------- Beta ------- #
            betaStr = ""
            for z in range(totalZONES):
                if (z not in data.dummyZONES) and (z not in data.termZONES):
                    for zp in nx.neighbors(data.zGraph, z):
                        if (z, zp) in beta_avg and len(beta_avg[(z, zp)]) > 0:
                            betaStr += "b"+str(z)+str(zp)+" : "+str(round(average(beta_avg[(z, zp)]), 4)) + " "
            ax.writeln("\n  Average Beta : ")
            ax.writeln("    "+betaStr)
            betaAvg2 = np.zeros((totalZONES, totalZONES))
            for z in range(totalZONES):
                if (z not in data.dummyZONES) and (z not in data.termZONES):
                    for zp in nx.neighbors(data.zGraph, z):
                        if len(beta_avg[(z, zp)]) > 0:
                            betaAvg2[z][zp] = average(beta_avg[(z, zp)])
            ax.writeln("\n  Runtime : " + str(round(runTime, 3)) + " Seconds")

        tEnd = ax.now()
        runTime += ax.getRuntime(tStart2, tEnd)

    ax.writeln("----------------------")
    max_return = max(buffer_Return)
    indx = np.where(buffer_Return==max_return)[0][0]
    max_vioPen = buffer_VioPenalty[indx]
    max_delayPen = buffer_DelayPenalty[indx]
    max_Vio = buffer_Vio[indx]
    max_Delay = buffer_Delay[indx]
    ax.writeln("\n\n  Best Episode Return : "+ str(max_return))
    ax.writeln("  Best Episode VioPenalty : " + str(max_vioPen))
    ax.writeln("  Best Episode DelayPenalty : " + str(max_delayPen))
    ax.writeln("  Best Res. Vio : " + str(max_Vio))
    ax.writeln("  Best Travel Delay : " + str(max_Delay))

    ax.writeln("----------------------------------------- EVALUATION ENDS ------------------------------------------")
    totalRunTime = ax.getRuntime(tStart, ax.now())
    ax.writeln("Total Runtime : " + str(round(totalRunTime, 3)) + " Seconds")
    ax.writeln("Evaluation Complete !")

def main():

    tStart = ax.now()
    runTime = 0
    init()
    data = env_data(mapName=MAP_ID)
    env = Maritime(data=data)
    agent = args.agent
    if agent == "random":
        mpa = randomAgent(load_model=LOAD_MODEL, dirName=dirName, data=data)
    elif agent == "tmin":
        mpa = tminAgent(load_model=LOAD_MODEL, dirName=dirName, data=data)
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
    for i_episode in range(1, EPISODES+1):

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
        epVioCountList = np.zeros(HORIZON+1)
        epTrDelayList = np.zeros(HORIZON+1)

        epVioPen = np.zeros(HORIZON+1)
        epDelayPen = np.zeros(HORIZON+1)


        epCount = 0

        # --------- Buffers --------- #
        buffer_nt_z = np.zeros((HORIZON, totalZONES))
        buffer_rt = np.zeros(HORIZON)
        buffer_rt_z = np.zeros((HORIZON, totalZONES))
        buffer_beta = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_ns = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_nsa = np.zeros((HORIZON, totalZONES, totalZONES, data.num_actions))        
        mpa.ep_init()

        
        for t in range(HORIZON):

            action_prob = mpa.getAction(t, cT, i_episode)
            buffer_nt_z[t] = cT.nt_z
            Rt, Rt_z, cT_new , vioPen, delayPen, beta = env.step(t, cT, action_prob)
            buffer_rt_z[t] = Rt_z
            buffer_rt[t] = Rt
            buffer_beta[t] = beta[0]
            buffer_ns[t] = cT_new.ns
            buffer_nsa[t] = cT_new.nsa


            # ------- Other Stats ------ #
            vioCount = np.sum(getVioCount(cT_new.nt_z, data.zGraph))
            trDelay = getTravelDelay(t, cT_new, data.zGraph)
            epRewardList[epCount] = Rt
            epVioCountList[epCount] = vioCount
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

        if i_episode % SHOULD_LOG == 0:
            ax.writeln("  " + str(t + 1) + " " + str("     ") + " " + str(cT_new.nt_z))


        # -------- Book Keeping ------- #
        mpa.storeRollouts(buffer_nt_z, buffer_rt_z, buffer_beta, buffer_rt)


        batchID += 1

        # ---------- Train Model --------- #
        if batchID % BATCH_SIZE == 0:
            mpa.train(i_episode)
            mpa.clear()
            batchID = 0
            mpa.epoch += 1

        epReward = getDiscountedReturn(epRewardList)
        epVioPenDiscount = round(getDiscountedReturn(epVioPen), 3)
        epDelayPenDiscount = round(getDiscountedReturn(epDelayPen), 3)

        buffer_runAvg[sampleID] = epReward
        buffer_VioPenalty[sampleID] = epVioPenDiscount
        buffer_DelayPenalty[sampleID] = epDelayPenDiscount

        buffer_vioCountAvg[sampleID] = np.sum(epVioCountList)
        buffer_trDelayAvg[sampleID] = np.sum(epTrDelayList)

        sampleID += 1

        # ----------- Save Model ----------- #
        if i_episode % SAVE_MODEL == 0:
            mpa.save_model()

        if runTime > TOTAL_RUNTIME:
            break

        # -------- Sample Average ------- #
        if i_episode % SAMPLE_AVG == 0:
            sampleID = 0

        # --------- Logs ------- #
        if i_episode % SHOULD_LOG == 0:
            epVioCount = np.sum(epVioCountList)
            epTrDelay = np.sum(epTrDelayList)

            ax.writeln("\n\n  Total Episode Reward : "+ str(round(epReward, 2)))
            ax.writeln("  Total Episode VioPenalty : " + str(round(epVioPenDiscount, 2)))
            ax.writeln("  Total Episode DelayPenalty : " + str(round(epDelayPenDiscount, 2)))

            ax.writeln("  ")
            ax.writeln("  Total Res. Vio : " + str(round(np.sum(epVioCountList), 2)))
            ax.writeln("  Total Travel Delay : " + str(round(np.sum(epTrDelayList), 2)))
            ax.writeln("  ")
            ax.writeln("  Avg. Res. Vio : " + str(round(np.average(epVioCountList), 2))+" std : "+str(round(np.std(epVioCountList), 2)))
            ax.writeln("  Avg. Travel Delay : " + str(round(np.average(epTrDelayList), 2))+" std : "+str(round(np.std(epTrDelayList), 2)))
            ax.writeln("  ")
            ax.writeln("  Max Res. Vio : " + str(round(np.max(epVioCountList), 2)))
            ax.writeln("  Max Travel Delay : " + str(round(np.max(epTrDelayList), 2)))


            # -------- Beta ------- #
            betaStr = ""
            for z in range(totalZONES):
                if (z not in data.dummyZONES) and (z not in data.termZONES):
                    for zp in nx.neighbors(data.zGraph, z):
                        if (z, zp) in beta_avg and len(beta_avg[(z, zp)]) > 0:
                            betaStr += "b"+str(z)+str(zp)+" : "+str(round(average(beta_avg[(z, zp)]), 4)) + " "
            ax.writeln("\n  Average Beta : ")
            ax.writeln("    "+betaStr)
            betaAvg2 = np.zeros((totalZONES, totalZONES))
            for z in range(totalZONES):
                if (z not in data.dummyZONES) and (z not in data.termZONES):
                    for zp in nx.neighbors(data.zGraph, z):
                        if len(beta_avg[(z, zp)]) > 0:
                            betaAvg2[z][zp] = average(beta_avg[(z, zp)])
            if i_episode % SAMPLE_AVG == 0:
                mpa.log(i_episode, epReward, betaAvg2, epVioCount, epTrDelay)
                os.system("cp -r runs/* ./log/"+dirName+"/plots/")
            # else:
            #     mpa.tensorboard(i_episode, epReward, betaAvg2, 0)
            ax.writeln("\n  Runtime : "+str(round(runTime, 3))+" Seconds")
        tEnd = ax.now()
        runTime += ax.getRuntime(tStart2, tEnd)


    ax.writeln("\n--------------------------------------------------")
    totalRunTime = ax.getRuntime(tStart, ax.now())
    ax.writeln("Total Runtime : " + str(round(totalRunTime, 3)) + " Seconds")
    ax.writeln("--------------------------------------------------")
    mpa.writer.close()

    # ----------- Evaluate
    if EVAL_EPISODES > 0:
        evaluate(data, env, mpa)


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

