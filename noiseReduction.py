from pathlib import Path
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from f1_calc import *
from scipy.interpolate import interp1d
import xlsxwriter


def mean_filtering(rssi_values):
    rssi_values[0] = median_filter(rssi_values[0], 20)

    return rssi_values

def gaussianfiltering(rssi_values):
    rssi_values[0] = gaussian_filter1d(rssi_values[0], 20)

    return rssi_values

def getStdRefactored(rssi_values):
    minTimeStamp = rssi_values[1].min()
    maxTimeStamp = rssi_values[1].max()

    times = np.arange(minTimeStamp, maxTimeStamp, 30)
    timeRange = 300



    stdValues = np.mean(np.lib.stride_tricks.sliding_window_view(rssi_values[0], (300)), axis = 1)
    stdValues = stdValues[stdValues > -45]
    # for t in times:
    #     values_to_consider = np.where((rssi_values[1] > t) & (rssi_values[1] < t + timeRange))
    #     if np.mean(rssi_values[0, values_to_consider]) > -50:
    #         stdValues.append(np.std(rssi_values[0, values_to_consider]))
    min = 0
    if len(stdValues) != 0:
        min = 1
    return min

# starting point loading in all pkl files
prox_files = [i for i in (Path.cwd()).glob("midges/*/*/*proximity.pkl")]
id_prox = {}

workbook = xlsxwriter.Workbook('arrays.xlsx')
worksheet = workbook.add_worksheet()


trues = pd.read_csv("seg2.csv", header = None)
trues2 = pd.read_csv("seg3.csv", header=None)
trues = trues.append(trues2)

startTime = pd.Timestamp(2019, 10, 24, 17, 3, 36, int(1000000 * 58 / 59.94))
startTime2 = pd.Timestamp(2019, 10, 24, 17, 7, 14, int(1000000 * 58 / 59.94))

affinityMatrix = np.full((51, 51), -1)
allMidges = []

for midge_path in prox_files:
    # the midge we are considering atm
    curid_str = midge_path.parent.parent.name
    print(curid_str)
    with open(midge_path, "rb") as m:
        midge_prox = pkl.load(m)
        midge_prox = midge_prox.sort_values("time")
        midge_prox = midge_prox[(midge_prox["id"] != 65535) & (midge_prox["id"] < 51)]

        for i in range(1, 50):
            # print(i)
            i = 30
            if i == 38 or i == int(curid_str) or i == 17 or int(curid_str) == 17:
                continue

            rssi_values = midge_prox[midge_prox["id"] == i]
            rssi_values = rssi_values[rssi_values["time"] > startTime]
            rssi_values["time"] = rssi_values["time"] - np.min(rssi_values["time"])
            rssi_values["time"] = rssi_values["time"].apply(lambda x : x.total_seconds())

            if len(rssi_values) == 0:
                continue
            interp = interp1d(rssi_values["time"], rssi_values["rssi"])
            whole_seconds = np.arange(0, int(np.max(rssi_values["time"])), 1)
            interpolatedValues = interp(whole_seconds)

            rssi_values = np.array([interpolatedValues, whole_seconds])
            # plt.plot(rssi_values[0])
            # plt.title("Normal values "+ str(i))
            # plt.show()

            medianFiltered = mean_filtering(rssi_values)
            # plt.plot(medianFiltered[0])
            # plt.title("Median filter " + str(i))
            # plt.show()

            gaussianFiltered = gaussianfiltering(medianFiltered)
            # plt.plot(gaussianFiltered[0])
            # plt.title("Gaussian filter "+str(i))
            # plt.show()

            std = getStdRefactored(gaussianFiltered)

            # if affinityMatrix[int(curid_str)][i] != 20:
            #     if std < affinityMatrix[int(curid_str)][i]:
            #         affinityMatrix[int(curid_str)][i] = std
            # else:
            affinityMatrix[int(curid_str)][i] = std
            row = i
            for col, data in enumerate(affinityMatrix):
                worksheet.write_column(row, col, data)


    allMidges.append(int(curid_str))
print(affinityMatrix)
workbook.close()
groups = iterate_climb_learned(affinityMatrix, len(allMidges))

print(groups)




