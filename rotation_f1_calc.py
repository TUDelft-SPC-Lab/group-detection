# -*- coding: utf-8 -*-
"""
Created on Wen May 4 ‏‎19:22:35 2022

@author: Ardy Zwanenburg uses code from proximity_f1_calc, dominant_sets and  f1_calc
"""

import numpy as np
from pathlib import Path
import pandas as pd
import pickle as pkl
import re

# for a set of vectors of the form [0,1,0,...,1], return a set of vectors of group names, using dict
# Made by Bram Dikker
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
# Made by Bram Dikker
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

def f1_calc(threshold, max_timeout, op='min'):
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
            print(id_prox[ii])

        sorted_id_prox = sorted(id_latestprox.items())

        matrix = []
        id_matrix_pos = {}

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

        groups = iterate_climb_learned(matrix.A, len(allMidges))
        groups = group_names(groups, len(allMidges), id_matrix_pos)
        TP_n, FN_n, FP_n, precision, recall = f1.group_correctness(groups, truegroups, 1, non_reusable=False)
        avg_results += np.array([precision, recall])
        print(str(time) + " annotated line: " + str(i + 1) + "/" + str(len(trues.index)) + " results " + str(correctness))
        # print(groups)

    avg_results /= counter

    if avg_results[0] * avg_results[1] == 0:
        f1_avg = 0
    else:
        f1_avg = float(2) * avg_results[0] * avg_results[1] / (avg_results[0] + avg_results[1])
    
    return f1_avg, avg_results[0], avg_results[1], counter, discarded_total

# Get all file locations for rotation pkl files
prox_files = [i for i in (Path.cwd()).glob("data/*/*/*rotation.pkl")]

# Reading the groundtruth
trues = pd.read_csv("FF/seg2.csv", header=None)
trues2 = pd.read_csv("FF/seg3.csv", header=None)
trues = trues.append(trues2)

# Setting the times
startTime = pd.Timestamp(2019, 10, 24, 17, 3, 36, int(1000000 * 58 / 59.94))
startTime2 = pd.Timestamp(2019, 10, 24, 17, 7, 14, int(1000000 * 58 / 59.94))

# Fetch all midge files
id_prox = {}
for midge_path in prox_files:
    curid_str = midge_path.parent.parent.name
    with open(midge_path, "rb") as m:
        midge_prox = pkl.load(m)
        midge_prox = midge_prox.set_index("time")
    id_prox[int(curid_str)] = midge_prox.between_time((startTime - pd.Timedelta(pd.offsets.Second(61))).time(), (
            startTime + pd.Timedelta(pd.offsets.Second(len(trues) + 61))).time())

print(f1_calc(-55, 30, 'avg'))