#Started from May 25th, 2022. Written by Park JongHyeon
#This is a code that modified Machine B to perform multi-process calculations using MPI. Modification start date: 2024/01/20
#file I/O reduction version. Modification start date: 2024/07/23
#한 번에 대량 입력 가능하도록 수정함. PKA info 입력부분을 제외하면 240816과 차이 없음

import subprocess
import time, random
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from mpi4py import MPI

#MPI initialize
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#get PKA information that we want to analzye
def get_PKA_info_from_file(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True)

    df_original = df.iloc[:, [1,2,3,4]].copy()
    df_original.columns = ['energy', 'dir_x', 'dir_y', 'dir_z']
    df_original.round(3)

    df_processed = df_original.copy()
    df_processed['dir_x'] = df_processed['dir_x'].abs()
    df_processed['dir_y'] = df_processed['dir_y'].abs()
    df_processed['dir_z'] = df_processed['dir_z'].abs()
    df_processed = df_processed.round(3)

    df_processed[['dir_x', 'dir_y', 'dir_z']] = np.sort(df_processed[['dir_x', 'dir_y', 'dir_z']], axis=1)[:, ::-1]

    return df_processed, df_original

#execute Machine A1 and get defect number - without file I/O
def run_A1(dataframe, PKA_index):
    new_list = dataframe.iloc[PKA_index].values.tolist() #0:ene 1:x_dir 2:y_dir 3:z_dir
    A1_int = random.randint(1,10000) + rank*119131
    commandA1 = "./def_number.exe {} {} {} {} {}".format(new_list[0], new_list[1], new_list[2], new_list[3], A1_int)
    result = subprocess.run(commandA1, capture_output=True, text=True, shell=True)
    return int(result.stdout.strip())

#execute Machine A2 and get defect position - without file I/O
def run_A2(dataframe, PKA_index, defnum):
    new_list = dataframe.iloc[PKA_index].values.tolist() #0:ene 1:x_dir 2:y_dir 3:z_dir
    A2_int = random.randint(1,10000) + rank*119131
    testname = '%s'%PKA_index
    commandA2 = "./def_config.exe {} {} {} {} {} {} {}".format(testname, new_list[0], new_list[1], new_list[2], new_list[3], A2_int, defnum)
    process = subprocess.run(commandA2, stdout=subprocess.PIPE, text=True, shell=True)

    output_data = []
    visual_data = []
    current_output = []
    for line in process.stdout.splitlines():
        line = line.strip()
        if line == "FLAG_cut":
            output_data.append(current_output)
            current_output = []
        elif line == "FLAG_visual":
            visual_data.append(current_output)
            current_output = []
        else:
            current_output.append(line)

    #process.wait() #일단 지우기는 하는데 프로세스 안정화를 위해 필요할 수 있으니 확인 바람

    parsed_data = []
    for block in output_data:
        parsed_block = [list(map(float, line.split())) for line in block]
        parsed_data.append(parsed_block)
    
    parsed_data = np.array(parsed_data)
    return parsed_data, visual_data

#convert A2 result (3, 15625, 4) to numpy (6,125,125)
def np_convert(cfg):
    #inputnp[0] : xy, inputnp[1] : yz, inputnp[2] : xz
    #inputnp[*][:,2]: sia, inputnp[*][:,3]: vac
    xy_v = cfg[0][:,3].reshape(-1,125)
    yz_v = cfg[1][:,3].reshape(-1,125)
    xz_v = cfg[2][:,3].reshape(-1,125)

    xy_s = cfg[0][:,2].reshape(-1,125)
    yz_s = cfg[1][:,2].reshape(-1,125)
    xz_s = cfg[2][:,2].reshape(-1,125)

    stack_array = np.stack((xy_v, yz_v, xz_v, xy_s, yz_s, xz_s), axis=2) # axis 0은 (6,125,125), axis 2는 (125,125,6)
    stack_array = np.expand_dims(stack_array, axis=0)
    return stack_array

#Machine B part
def run_B(cfg_stack):
    m_pred = machine_model.predict(cfg_stack, verbose=0)
    class_pred = np.argmax(m_pred, axis=1)
    return int(class_pred)

#Calculation time check
def cal_info_write(energy, iteration, start_time):
    end_time = time.time()
    total_time = end_time - start_time
    with open('./computing_time_%s.dat'%index, 'w') as tfile:
        tfile.write(str(energy))
        tfile.write("\t")
        tfile.write(str(iteration))
        tfile.write("\t")
        tfile.write(str(round(total_time,3)))
    return 0

#Get PKA information and load the machine
if rank == 0:
    PKA_table, PKA_table_original = get_PKA_info_from_file("./30keV_rand.txt")
    PKA_list_num = len(PKA_table)
    machine_model = load_model('./Final_machine_211109_3.h5')
else:
    PKA_table, PKA_table_origianl = None, None
    PKA_list_num = None
    machine_model = None

#Broadcase PKA_table and MachineB to all processes
PKA_table = comm.bcast(PKA_table, root=0)
PKA_list_num = comm.bcast(PKA_list_num, root=0)
machine_model = comm.bcast(machine_model, root=0)


#Parallel loop for each index in PKA_table
iteration_limit = 1000

for index in range(PKA_list_num):
    found_yes = False
    found_yes_rank = None
    start_time = time.time() #시작시간 기록

    #if found_yes:
    #    break

    if rank == 0:
        def_num = run_A1(PKA_table, index)
        if def_num == 0: # 결함 수가 0인 경우 즉시 stop하고 빈 visual file을 생성
            cal_info_write('temp', 1, start_time)
            with open('./primary_damage_structure_{}.dat'.format(index), 'w') as vfile:
                vfile.write('no defect')
            break
    else:
        def_num = None
    
    def_num = comm.bcast(def_num, root=0)
    iter = 0

    while not found_yes:
        iter += 1
        def_config, visual_data  = run_A2(PKA_table, index, def_num)
        def_proj = np_convert(def_config)
        cfg_type = run_B(def_proj) #0은 No, 1은 Yes

        if cfg_type > 0.9: # yes 판정이 될 경우
            found_yes = True
            found_yes_rank = rank # 현재 프로세스의 랭크를 저장

        if found_yes_rank is not None:
            found_yes_rank = comm.bcast(found_yes_rank, root=found_yes_rank)
            if rank == found_yes_rank: # Yes를 발견한 프로세스만 저장 수행
                with open('./primary_damage_structure_{}.dat'.format(index), 'w') as vfile:
                    for sublist in visual_data:
                        vfile.write('\n'.join(map(str,sublist)))
                cal_info_write('temp', iter, start_time)
                print("Process {} found the index {} yes structure \n".format(rank, index))
            break
        
        if iter > iteration_limit:
            cal_info_write('temp', iter, start_time)
            print("calculation over {}, stop working".format(iteration_limit))
            with open('./primary_damage_structure_{}.dat'.format(index), 'w') as vfile:
                vfile.write('over iteration number {}'.format(iteration_limit))
            break
    comm.Barrier()
    time.sleep(2)