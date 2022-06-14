# -*- coding: utf-8 -*-
"""
Created on Wensday May 25 11:31:28 2022

@author: Ardy Zwanenburg uses code from proximity_f1_calc.py and Quaternion.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle as pkl


def load_data(trues):    
    rotation_files = [i for i in (Path.cwd()).glob("data/*/*/*rotation_euler.pkl")]
    id_rotation = {}

    startTime = pd.Timestamp(2019, 10, 24, 17, 3, 36, int(1000000 * 58 / 59.94))
    startTime2 = pd.Timestamp(2019, 10, 24, 17, 7, 14, int(1000000 * 58 / 59.94))
    for midge_path in rotation_files:
        curid_str = midge_path.parent.parent.name
        with open(midge_path, "rb") as m:
            midge_rot = pkl.load(m)
            midge_rot = midge_rot.set_index("time")
        
        id_rotation[int(curid_str)] = midge_rot.between_time((startTime - pd.Timedelta(pd.offsets.Second(61))).time(), (
                startTime + pd.Timedelta(pd.offsets.Second(len(trues) + 61))).time())
    return id_rotation

def compare_rotation(a, b, orientation):
    # Set the range to 0-360
    a += 180
    b += 180

    interval_a = [(a-orientation)%360, (a+orientation)%360]
    interval_b = [(b-orientation)%360, (b+orientation)%360]

    if (interval_a[0] < interval_a[1]):
        if (interval_b[0] < interval_b[1]):
            if (interval_a[0] < interval_b[0] < interval_a[1]) or (interval_a[0] < interval_b[1] < interval_a[1]):
                #print("Possible facing v1")
                return False
        else:
            if (interval_a[0] < interval_b[1]) or (interval_a[1] > interval_b[0]):
                #print("Possible facing v2")
                return False
    else:
        if (interval_b[0] < interval_b[1]):
            if (interval_a[0] < interval_b[1]) or (interval_a[1] > interval_b[0]):
                #print("Possible facing v3")
                return False
        else:
            #print("Possible facing v4")
            return False
    #if (interval_b[1] <= interval_a[0] or interval_b[0] >= interval_a[1]):
        #print("Found opposite angles")
        #return True
    return True


def rotation_filter(matrix, trues, allMidges, time, orientation):
    matrix = np.array(matrix)
    id_rotation = load_data(trues)
    index_y, index_x = np.where(matrix > 0)
    store = {}
    for y, x in zip(index_y, index_x):
        #try:
            z1 = x + 1
            z2 = y + 1
            if x >= 37:
                z1 += 1
            if y >= 37:
                z2 += 1
            if z1 == 39 or z2 == 39:
                continue
            if y not in store:
                # Find the closes time to the given time
                abs_deltas_from_target_date = np.absolute(id_rotation[z2].index - time.to_datetime64())
                index_time_y = np.argmin(abs_deltas_from_target_date)
                store[y] = id_rotation[z2]["Y"].iloc[index_time_y]
        
            if x not in store:
                # Find the closes time to the given time
                abs_deltas_from_target_date = np.absolute(id_rotation[z1].index - time.to_datetime64())
                index_time_x = np.argmin(abs_deltas_from_target_date)
                store[x] = id_rotation[z1]["Y"].iloc[index_time_x]

            if (compare_rotation(store[x], store[y], orientation)):
                print("Impossibilty to see each other")
                print(store[x], store[y], orientation)
                matrix[y][x] = 0.0
        # except:
            #print("=== Error happend continue to the next one ===")
            #continue
    return np.asmatrix(matrix)

