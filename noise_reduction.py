from pathlib import Path
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from scipy.ndimage import gaussian_filter1d, median_filter
import datetime

def find_Max_Timestamp(rssi_values):
    max_value = None
    for num in rssi_values:
        if (max_value is None or num[1] > max_value):
            max_value = num[1]

    return max_value

def find_Min_Timestamp(rssi_values):
    min_value = None
    for num in rssi_values:
        if (min_value is None or num[1] < min_value):
            min_value = num[1]

    return min_value
def plot_values(rssi_values, minTime, maxTime):

    y, x = zip(*rssi_values)
    plt.plot(x, y, linestyle='', marker='o', color='b')
    plt.title("Midge:" + curid_str + " Detected midge" + str(i))
    plt.xlim((minTime,maxTime))

    plt.show()

def to_NPArray(rssi_values):
    filtered = []
    for item in rssi_values:
        filtered.append([item[0], item[1].value])

    filtered = np.array(filtered).T
    return filtered

    # go over all rssi values
    # consider a certain window size
    # take the mean of the window and replace the value with this value
def mean_filtering(rssi_values):

    rssi_values[0] = median_filter(rssi_values[0], 5)


    return rssi_values


def gaussianfiltering(rssi_values):
    # filtered[0] = np.convolve(np.array([0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]), filtered[0], mode='same')
    rssi_values[0] = gaussian_filter1d(rssi_values[0], 5)

    return rssi_values






def getTimeIntervals(rssi_values):
    startTime = find_Min_Timestamp(rssi_values)
    endTime = find_Max_Timestamp(rssi_values)
    return pd.interval_range(start=startTime, end=endTime, periods=30)

# loops with a certain time window through all values and calculates their std
# based on that we can see when and if two midges might be in a conversation group
def getStdOfTimeInterval(mean_array):
    endTime = max(mean_array[1])
    for x in range(0, len(mean_array[1])-1):
        startWindow = mean_array[1][x]

        # TODO: right now endwindow will always be smaller than endtime, need to add 5 minutes
        endWindow = mean_array[1][x]
        # + timedelta(minutes=5)

        if endWindow <= endTime:
            values = []
            j = x
            diff = 0
            # TODO: is this window correct? -> check
            while (pd.Timestamp(diff).minute < 10):
                if j == len(mean_array[1]):
                    break
                else:
                    if mean_array[0][j] > -50:
                        values.append(mean_array[0][j])
                        diff = mean_array[1][j] - startWindow
                        j += 1
                    else:
                        j += 1

            std = np.std(values)
            if(std < 0.5):
                if(pd.Timestamp(diff).minute == 10):
                    print('midge', curid_str, 'detected', i, std, 'timetamp', pd.Timestamp(diff).minute)
        else:
            break



# starting point loading in all pkl files
prox_files = [i for i in (Path.cwd()).glob("midges/*/*/*proximity.pkl")]


for midge_path in prox_files:
    # the midge we are considering atm
    curid_str = midge_path.parent.parent.name
    print(curid_str)
    with open(midge_path, "rb") as m:
        midge_prox = pkl.load(m)
        midge_prox = midge_prox.sort_values("time")
        # midge_prox = midge_prox.set_index("time")
        midge_prox = midge_prox[(midge_prox["id"] != 65535) & (midge_prox["id"] < 51)]

        for i in range (2, 50):
            # print(2)
            # there is no data for device 38
            if i == 38:
                continue
            rssi_values = []
            for idx,row in midge_prox.iterrows():
                if row["id"] == i and i != int(curid_str):
                    if row["time"] in rssi_values:
                        continue
                    else:
                        rssi_values.append(tuple((row["rssi"], row["time"])))

            minTime = find_Min_Timestamp(rssi_values)
            maxTime = find_Max_Timestamp(rssi_values)

            rssi_values = to_NPArray(rssi_values)
            # plt.scatter(rssi_values[1], rssi_values[0])
            # plt.title('rssi' + curid_str + str(i))
            # plt.show()


            medianFiltered = mean_filtering(rssi_values)
            # plt.scatter(medianFiltered[1], medianFiltered[0])
            # plt.title('median' +  curid_str + str(i))
            # plt.show()
            gaussianFiltered = gaussianfiltering(medianFiltered)
            plt.scatter(gaussianFiltered[1], gaussianFiltered[0])
            plt.title('gaussian' + curid_str + str(i))
            plt.show()


            getStdOfTimeInterval(gaussianFiltered)










