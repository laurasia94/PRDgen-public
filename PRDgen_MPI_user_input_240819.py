#Started from May 25th, 2022. Written by Park JongHyeon
#This is a code that modified Machine B to perform multi-process calculations using MPI. Modification start date: 2024/01/20
#file I/O reduction version. Modification start date: 2024/07/23
#PRDgen_MPI_240816.py의 계보를 이어갑니다. 입력 방식은 user input

import PRDgen_modules as PRD
import time
from tensorflow.keras.models import load_model
from mpi4py import MPI

#MPI initialize
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Get PKA information
if rank == 0:
    PKA_table, PKA_table_original = PRD.get_PKA_info()
    PKA_list_num = len(PKA_table)
else:
    PKA_table, PKA_table_origianl = None, None
    PKA_list_num = None

#Broadcase PKA_table to all processes
PKA_table = comm.bcast(PKA_table, root=0)
PKA_list_num = comm.bcast(PKA_list_num, root=0)

#Load model
CNN_model = load_model('./Final_machine_211109_3.h5')

#Parallel loop for each index in PKA_table
iteration_limit = 1000
for index in range(PKA_list_num):
    found_yes = False
    found_yes_rank = None
    start_time = time.time() #시작시간 기록

    if rank == 0:
        def_num = PRD.run_A1(PKA_table, index)
        if def_num == 0: # 결함 수가 0인 경우 즉시 stop하고 빈 visual file을 생성
            PRD.cal_info_write(index, 1, start_time)
            with open('./primary_damage_structure_{}.dat'.format(index), 'w') as vfile:
                vfile.write('no defect')
            found_yes = True
    else:
        def_num = None
    
    def_num = comm.bcast(def_num, root=0)
    found_yes = comm.bcast(found_yes, root=0)

    if found_yes:
        comm.Barrier()
        continue

    iter = 0
    while not found_yes:
        iter += 1
        def_config, visual_data  = PRD.run_A2(PKA_table, index, def_num, rank)
        print("rank{}_iteration{}".format(rank, iter))
        def_proj = PRD.np_convert(def_config)
        cfg_type = PRD.run_B(CNN_model, def_proj) #0은 No, 1은 Yes

        if cfg_type > 0.9: # Yes 판정이 될 경우 visual file을 저장하고 stop
            found_yes = True
            found_yes_rank = rank # 현재 프로세스의 랭크를 저장
        
        found_yes_rank = comm.bcast(found_yes_rank, root=rank)

        if found_yes_rank is not None and rank == found_yes_rank:
            with open('./primary_damage_structure_{}.dat'.format(index), 'w') as vfile:
                for sublist in visual_data:
                    vfile.write('\n'.join(map(str,sublist)))
            PRD.cal_info_write(index, iter, start_time)
            print("Process {} found the index {} yes structure \n".format(rank, index))
            break

        if iter > iteration_limit:
            PRD.cal_info_write(index, iter, start_time)
            print("calculation over {}, stop working".format(iteration_limit))
            with open('./primary_damage_structure_{}.dat'.format(index), 'w') as vfile:
                vfile.write('over iteration number {}'.format(iteration_limit))
            break

    comm.Barrier()
    time.sleep(2)