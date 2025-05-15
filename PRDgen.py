import sys
import math
import random
import subprocess
import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load model
CNN_model = load_model('./Final_machine_211109_3.h5')

def preprocess_direction(x, y, z):
    pos_values = [abs(float(x)), abs(float(y)), abs(float(z))]
    sorted_values = sorted(pos_values, reverse=True)
    rounded = [round(val, 3) for val in sorted_values]
    norm = math.sqrt(sum(v**2 for v in rounded))
    normalized = [round(v / norm, 3) for v in rounded]
    return normalized

def get_PKA_info(energy, dir_x, dir_y, dir_z):
    directions = sorted([abs(dir_x), abs(dir_y), abs(dir_z)], reverse=True)
    df_processed = pd.DataFrame([{
        'energy': round(energy, 3),
        'dir_x': directions[0],
        'dir_y': directions[1],
        'dir_z': directions[2]
    }])
    return df_processed

def run_A1(df):
    e, x, y, z = df.iloc[0]
    rand_seed = random.randint(1, 10000)
    cmd = f"./def_number.exe {e} {x} {y} {z} {rand_seed}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())

def run_A2(df, defnum):
    e, x, y, z = df.iloc[0]
    rand_seed = random.randint(1, 10000) + 119131
    cmd = f"./def_config.exe result {e} {x} {y} {z} {rand_seed} {defnum}"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, shell=True)

    output_data, visual_data, current_output = [], [], []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line == "FLAG_cut":
            output_data.append(current_output)
            current_output = []
        elif line == "FLAG_visual":
            visual_data.append(current_output)
            current_output = []
        else:
            current_output.append(line)

    parsed_data = [
        [list(map(float, line.split())) for line in block]
        for block in output_data
    ]
    return np.array(parsed_data), visual_data

def np_convert(cfg):
    xy_v = cfg[0][:,3].reshape(-1,125)
    yz_v = cfg[1][:,3].reshape(-1,125)
    xz_v = cfg[2][:,3].reshape(-1,125)
    xy_s = cfg[0][:,2].reshape(-1,125)
    yz_s = cfg[1][:,2].reshape(-1,125)
    xz_s = cfg[2][:,2].reshape(-1,125)
    stack = np.stack((xy_v, yz_v, xz_v, xy_s, yz_s, xz_s), axis=2)
    return np.expand_dims(stack, axis=0)

def run_B(model, cfg_stack):
    if cfg_stack.size == 0:
        print("Empty defect config.")
        return 0, 0.0
    pred = model.predict(cfg_stack, verbose=0)
    return int(np.argmax(pred, axis=1)), round(pred[0][1] * 100, 2)

def PRDgen(energy, direction):
    dir_x = direction[0]
    dir_y = direction[1]
    dir_z = direction[2]
    df = get_PKA_info(energy, dir_x, dir_y, dir_z)
    start_time = time.time()
    max_iter = 1000
    iteration = 0

    def_num = run_A1(df)
    if def_num == 0:
        with open("primary_damage_structure_0.dat", "w") as f:
            f.write("no defect")
        return def_num, 0.0, 0, 0.0

    while iteration < max_iter:
        iteration += 1
        def_cfg, visual_data = run_A2(df, def_num)
        def_proj = np_convert(def_cfg)
        cfg_type, score = run_B(CNN_model, def_proj)

        if cfg_type > 0.9:
            with open("primary_damage_structure.dat", "w") as f:
                for sub in visual_data:
                    f.write('\n'.join(sub) + '\n')
            break
    else:
        with open("primary_damage_structure.dat", "w") as f:
            f.write("over iteration number 1000")

    total_time = time.time() - start_time
    return def_num, round(total_time, 3), iteration, score

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python PRDgen_250507_linux energy(keV) dir_x dir_y dir_z")
        sys.exit(1)

    energy = float(sys.argv[1])
    dir_x = float(sys.argv[2])
    dir_y = float(sys.argv[3])
    dir_z = float(sys.argv[4])

    direction = preprocess_direction(dir_x, dir_y, dir_z)
    def_num, t_time, itr, score = PRDgen(energy, direction)
    print("DefectNum: ", def_num)
    print("Calculation time: ", t_time)
    print("defect structure saved as 'primary_damage_structure.dat'")