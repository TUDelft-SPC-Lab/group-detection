# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:23:23 2022

@author: Bram Dikker uses code from dominant_sets and  f1_calc
"""

import pickle as pkl
import pandas as pd
import f1_calc as f1
import re
from pathlib import Path
from dominant_sets import *
from rotation_filter import *


# for a set of vectors of the form [0,1,0,...,1], return a set of vectors of group names, using dict
def group_names(bool_groups, n_people, matrix_pos_id):
    groups = []
    for bool_group in bool_groups:
        group = []
        for i in range(n_people):
            if (bool_group[i]):
                group.append("ID_00" + str(matrix_pos_id[i]))
        groups.append(group)
    return groups


# Find groundtruths and included midges
def findTrueVal(offset, df):
    truth = df.iloc[offset].to_numpy()
    truth = truth[1]
    text = re.findall("\<(.*?)\>", truth)
    truegroups = []
    allMidges = []
    for group in text:
        newgroup = []
        for midge in re.findall('\d+', group):
            newgroup.append("ID_00" + midge)
            allMidges.append(int(midge))
        truegroups.append(newgroup)
    return truegroups, allMidges


# Find latest rssi, depending on time and what midges are inlcuded and depending on the maximal time difference
def findRSSI(df, curid, time, allNeighbours, max_timeout):
    prox_near = df
    near_ids = allNeighbours.copy()
    near_ids.remove(curid)
    rssi_at_time = []
    discared_count = 0
    for n_id in near_ids:
        # cur_near = prox_near[prox_near["id"] == n_id]
        # using masks
        mask = prox_near['id'].values == n_id
        cur_near = prox_near.values[mask]
        cur_near_time = prox_near.index.values[mask]
        # cur_near = prox_near.query('id == @n_id')
        if (cur_near.size == 0):
            rssi_at_time.append([n_id, -100])
        else:
            # Use vectorization with numpy, is about 4x faster
            times = cur_near_time
            abs_deltas_from_target_date = np.absolute(times - time.to_datetime64())
            index_of_min_delta_from_target_date = np.argmin(abs_deltas_from_target_date)
            if abs_deltas_from_target_date[index_of_min_delta_from_target_date] > max_timeout:
                rssi_at_time.append([n_id, -100])
                discared_count += 1
                continue
            # index_closest = cur_near.index.get_loc(time, method='nearest')
            rssi_at_time.append([n_id, cur_near[index_of_min_delta_from_target_date][1]])
            # print(abs((pd.Timestamp(entry_closest.name) - pd.Timestamp(time))/np.timedelta64(1, 's')))
            # print((pd.Timestamp(entry_closest.name) - pd.Timestamp(time))/np.timedelta64(1, 's'))
        # print(cur_near.iloc[cur_near.index.get_loc(time, method='nearest')])
    return discared_count, np.array(rssi_at_time)


prox_files = [i for i in (Path.cwd()).glob("data/*/*/*proximity.pkl")]
id_prox = {}

trues = pd.read_csv("ff/seg2.csv", header=None)
trues2 = pd.read_csv("ff/seg3.csv", header=None)
trues = trues.append(trues2)

startTime = pd.Timestamp(2019, 10, 24, 17, 3, 36, int(1000000 * 58 / 59.94))
startTime2 = pd.Timestamp(2019, 10, 24, 17, 7, 14, int(1000000 * 58 / 59.94))
# Fetch all midge files
for midge_path in prox_files:
    curid_str = midge_path.parent.parent.name
    with open(midge_path, "rb") as m:
        midge_prox = pkl.load(m)
        midge_prox = midge_prox.sort_values("id")
        midge_prox = midge_prox.set_index("time")
        midge_prox = midge_prox[(midge_prox["id"] != 65535) & (midge_prox["id"] < 51)]
    id_prox[int(curid_str)] = midge_prox.between_time((startTime - pd.Timedelta(pd.offsets.Second(61))).time(), (
            startTime + pd.Timedelta(pd.offsets.Second(len(trues) + 61))).time())


# range(0, len(trues.index) - 1
def f1_calc(threshold, max_timeout, reconstruct=True, op='min', rotation=-1):
    avg_results = np.array([0.0, 0.0])
    counter = 0
    discarded_total = 0
    max_timeout = max_timeout * 1000000000

    for i in range(0, len(trues.index)):
        time = startTime + pd.Timedelta(pd.offsets.Second(i))
        truegroups, allMidges = findTrueVal(i, trues)
        if (len(allMidges) != len(set(allMidges))):
            print("Duplicate found on line " + str(i + 1))
            continue
        counter += 1
        allMidges = sorted(allMidges)
        id_latestprox = {}
        for ii in allMidges:
            # .between_time((time - pd.Timedelta(pd.offsets.Second(61))).time(), (time + pd.Timedelta(pd.offsets.Second(61))).time())
            discarded, latest_rssi = findRSSI(id_prox[ii], ii, time, allMidges, max_timeout)
            id_latestprox[ii] = latest_rssi
            discarded_total += discarded

        # Reconstruction of midge 17
        if reconstruct:
            if 17 in id_latestprox.keys():
                array = []
                for midge in allMidges:
                    for midge2, rssi in id_latestprox[midge]:
                        if midge2 == 17:
                            array.append([midge, rssi])
                id_latestprox[17] = np.array(array)

        sorted_id_prox = sorted(id_latestprox.items())

        matrix = []
        id_matrix_pos = {}

        # Threshold function, most interesting to edit
        def convert(x):
            # This only occurs if Midge 17 is not reconstructed as it reported back all rssi values as 1
            if (x == 1):
                return 0.0
            if (x >= threshold):
                return 1.0
            return 0.0

        # Build the matrix row by row
        for ii, prox in enumerate(sorted_id_prox):
            row = prox[1][:, 1]
            # row = list(map(convert,row))
            # insert zero to self
            row = np.insert(row, ii, 0.0)
            matrix.append(row)
            # Remember midge position in matrix
            id_matrix_pos[ii] = prox[0]

        matrix = np.asmatrix(matrix)
        # print(matrix)
        matrix = symmetrize_A(matrix, op)
        # print(matrix)
        matrix = np.vectorize(convert)(matrix)
        # print(matrix)
        # Fill the diagonal again
        np.fill_diagonal(matrix, 0.0)

        if rotation != -1:
            matrix = rotation_filter(matrix, trues, allMidges, time, rotation)

        groups = iterate_climb_learned(matrix.A, len(allMidges))
        groups = group_names(groups, len(allMidges), id_matrix_pos)
        correctness = f1.group_correctness(groups, truegroups, 1, non_reusable=False)
        TP_n, FN_n, FP_n, precision, recall = correctness
        avg_results += np.array([precision, recall])
        print(str(time) + " annotated line: " + str(i + 1) + "/" + str(len(trues.index)) + " results " + str(correctness))
        # print(groups)

    avg_results /= counter

    if avg_results[0] * avg_results[1] == 0:
        f1_avg = 0
    else:
        f1_avg = float(2) * avg_results[0] * avg_results[1] / (avg_results[0] + avg_results[1])
    # print(counter)
    # print(discarded_total)
    # print(f1_avg)
    # print(avg_results[0])
    # print(avg_results[1])
    # print(time)
    return f1_avg, avg_results[0], avg_results[1], counter, discarded_total


def get_results():
    results = []
    threshold = [-55]
    timeout = [10, 20, 30, 40, 50, 60]
    operators = ['min', 'max', 'avg']
    reconstruction = [True, False]
    for tresh in threshold:
        for tim in timeout:
            for oper in operators:
                for rec in reconstruction:
                    f_1avg, precision, recall, items, discarded = f1_calc(tresh, tim, op=oper, reconstruct=rec)
                    results.append([tresh, tim, oper, rec, f_1avg, precision, recall, items, discarded])
    dfresults = pd.DataFrame(results,
                             columns=['Threshold', 'Timeout', 'Symmetrisation', 'Reconstruction', 'F1 measure',
                                      'Precision',
                                      'Recall', 'Items', 'Timeouts occurred'])
    dfresults.to_excel("Reults.xlsx", sheet_name='F-measure')
    print(results)

def get_results_Ardy():
    results = []
    rotation = [-1, 10, 20, 30, 35, 45, 63, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    for rot in rotation:
        f_1avg, precision, recall, items, discarded = f1_calc(-55, 30, op='avg', reconstruct=True, rotation=rot)
        results.append([-55, 30, 'avg', True, rot, f_1avg, precision, recall, items, discarded])
    dfresults = pd.DataFrame(results,
                             columns=['Threshold', 'Timeout', 'Symmetrisation', 'Reconstruction', 'orientation_angle',
                                      'F1 measure', 'Precision',
                                      'Recall', 'Items', 'Timeouts occurred'])
    dfresults.to_excel("Reults.xlsx", sheet_name='F-measure')
    print(results)


print(get_results_Ardy())
