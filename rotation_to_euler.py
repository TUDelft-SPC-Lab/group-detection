import pandas as pd
from pathlib import Path
import pickle as pkl
from Quaternion import *

rotation_files = [i for i in (Path.cwd()).glob("data/*/*/*rotation.pkl")]
id_rotation = {}

startTime = pd.Timestamp(2019, 10, 24, 17, 3, 36, int(1000000 * 58 / 59.94))
startTime2 = pd.Timestamp(2019, 10, 24, 17, 7, 14, int(1000000 * 58 / 59.94))
for midge_path in rotation_files:
    new_path = str(midge_path).split(".")[0] + "_euler." + str(midge_path).split(".")[1]
    curid_str = midge_path.parent.parent.name
    time = []
    all_x_euler = []
    all_y_euler = []
    all_z_euler = []
    with open(midge_path, "rb") as m:
        midge_rot = pkl.load(m)
        # midge_rot = midge_rot.set_index("time")
    for index in range(len(midge_rot['a'])):
        t = midge_rot['time'][index]
        a = midge_rot['a'][index]
        b = midge_rot['b'][index]
        c = midge_rot['c'][index]
        d = midge_rot['d'][index]
        quaternion = Quaternion(a, b, c, d)
        euler_x, euler_y, euler_z = quaternion.to_euler(degrees=True)
        time.append(t)
        all_x_euler.append(euler_x)
        all_y_euler.append(euler_y)
        all_z_euler.append(euler_z)

    print(all_y_euler)
    assert(len(time) == len(all_y_euler))
    df = pd.DataFrame(list(zip(time, all_x_euler, all_y_euler, all_z_euler)), columns=['time', 'X', 'Y', 'Z'])
    df.to_pickle(new_path)
    print("Midge", curid_str, "done")