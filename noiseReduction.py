from pathlib import Path
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from f1_calc import *
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz


def median_filtering(rssi_values):
    rssi_values[0] = median_filter(rssi_values[0], 30)

    return rssi_values


def gaussianfiltering(rssi_values):
    rssi_values[0] = gaussian_filter1d(rssi_values[0], 10)

    return rssi_values

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def getStdRefactored(rssi_values):
    windowValues = np.lib.stride_tricks.sliding_window_view(rssi_values[0], (240))

    stdValues = np.std(windowValues, axis=1)
    meanValues = np.mean(windowValues, axis=1)

    idx = np.where(meanValues > -55)
    stdValues = meanValues[meanValues > -55]
    # stdValues = np.std(stdValues)
    # idx = np.where(stdValues < 2)
    # stdValues = stdValues[stdValues < 2]

    # idx = np.where(stdValues < 1)
    # stdValues = stdValues[stdValues < 1]

    min = 0
    if len(stdValues) != 0:
        min = 1
    return min, idx


def findTrueVal(offset, df):
    truth = df.iloc[offset].to_numpy()
    truth = truth[1]
    text = re.findall("\<(.*?)\>", truth)
    truegroups = []
    allMidges = []
    for group in text:
        newgroup = []
        for midge in re.findall('\d+', group):
            # newgroup.append("ID_00" + midge)
            newgroup.append(int(midge))
            allMidges.append(int(midge))
        truegroups.append(newgroup)
    return truegroups, allMidges


def group_names(bool_groups, n_people, matrix_pos_id):
    groups = []
    for bool_group in bool_groups:
        group = []
        for i in range(n_people):
            if (bool_group[i]):
                # group.append("ID_00" + str(i))
                group.append(i)
        groups.append(group)
    return groups


def discardWrongMidges(trues, groups):
    for idx, group in enumerate(groups):
        if not any(idx in x for x in trues):
            group[group == 1] = 0
    return groups


def preprocess():
    # starting point loading in all pkl files
    prox_files = [i for i in (Path.cwd()).glob("midges/*/*/*proximity.pkl")]
    id_prox = {}

    conversationGroups = np.zeros((3000, 51, 51))
    conversationGroups[:, 0, 0] = range(0, 3000)

    recoverValuesFor17 = []

    trues = pd.read_csv("seg2.csv", header=None)
    trues2 = pd.read_csv("seg3.csv", header=None)
    trues = trues.append(trues2)

    # seg 2
    startTime = pd.Timestamp(2019, 10, 24, 17, 3, 36, int(1000000 * 58 / 59.94))

    # seg 3
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

                if i == 38 or i == int(curid_str) or int(curid_str) == 17 or int(curid_str) == 0:
                    continue

                rssi_values = midge_prox[midge_prox["id"] == i]
                rssi_values = rssi_values[rssi_values["time"] > startTime]
                rssi_values["time"] = rssi_values["time"] - np.min(rssi_values["time"])
                rssi_values["time"] = rssi_values["time"].apply(lambda x: x.total_seconds())

                if len(rssi_values) == 0:
                    continue
                interp = interp1d(rssi_values["time"], rssi_values["rssi"])
                whole_seconds = np.arange(0, int(np.max(rssi_values["time"])), 1)
                interpolatedValues = interp(whole_seconds)

                rssi_values = np.array([interpolatedValues, whole_seconds])
                rssi_values1 = np.array([interpolatedValues, whole_seconds])
                # plt.plot(rssi_values[0])
                # plt.title("Normal values "+ str(i))
                # plt.show()


                medianFiltered = median_filtering(rssi_values)
                # # plt.plot(medianFiltered[0])
                # # plt.title("Median filter " + str(i))
                # # plt.show()
                #
                gaussianFiltered = gaussianfiltering(rssi_values1)
                # plt.plot(gaussianFiltered[0])
                # plt.title("Gaussian filter "+str(i))
                # plt.show()
                order = 6
                fs = 50 # sample rate, Hz
                cutoff = 3.667  # desired cutoff frequency of the filter, Hz

                # Get the filter coefficients so we can check its frequency response.


                averageFiltered = butter_lowpass_filter(rssi_values, cutoff, fs, order)


                if i == 30:
                    plt.plot(averageFiltered[0][:1257], color = 'green', label = 'Low pass filter')
                    plt.plot(gaussianFiltered[0][:1257],  color = 'blue', label = "Gaussian filter")
                    plt.plot(medianFiltered[0][:1257], color ='red',label ='Median filter')
                    plt.title("Noise filters for "+ curid_str + " detecting "+ str(i))
                    plt.legend()
                    plt.xlabel('Time')
                    plt.ylabel('RSSI')
                    plt.show()
                # plt.plot(gaussianFiltered[0][:1257], color='blue', label="Gaussian filter")
                # plt.show()



                std, windowTimes = getStdRefactored(averageFiltered)
                # if len(windowTimes[0]) == 0:
                #     break
                for t in windowTimes[0]:
                    if t < 3000 :
                        conversationGroups[t, int(curid_str), i] = std

                # Recover midge 17
                if i == 17:
                    if t < 3000:
                        conversationGroups[t, i, int(curid_str)] = std

    return conversationGroups, trues


def grouping(conversationGroups, trues):
    avg_results = np.array([0.0, 0.0])
    for idx, group in enumerate(conversationGroups[:]):

        if (idx == 1276):
            break

        truegroups, allMidges = findTrueVal(idx, trues)
        for i in range(0, len(trues.index) - 1):
            if (len(allMidges) != len(set(allMidges))):
                print("Duplicate found on line " + str(i + 1))
                continue

        cleanGroups = discardWrongMidges(truegroups, group)
        cleanGroups = iterate_climb_learned(cleanGroups, 51)
        groups = group_names(cleanGroups, 51, conversationGroups[:])
        if groups[0][0] == 0:
            groups.pop(0)
        correctness = group_correctness(groups, truegroups, 2/3, non_reusable=False)
        TP_n, FN_n, FP_n, precision, recall = correctness
        avg_results += np.array([precision, recall])

        print(idx, groups)
        print('correctnes', str(correctness))

    avg_results /= len(trues)

    if avg_results[0] * avg_results[1] == 0:
        f1_avg = 0
    else:
        f1_avg = float(2) * avg_results[0] * avg_results[1] / (avg_results[0] + avg_results[1])

    print(f1_avg, avg_results[0], avg_results[1])


def run():
    conversationGroups, trues = None, None
    # try:
    #     conversationGroups = np.load('conversation_groups.npy', allow_pickle=True)
    #     trues = pd.read_csv('trues.csv')
    # except:
    #     conversationGroups, trues = preprocess()
    #     np.save('conversation_groups.npy', conversationGroups)
    #     trues.to_csv('trues.csv')

    conversationGroups, trues = preprocess()
    grouping(conversationGroups, trues)


run()
