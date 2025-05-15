# PRDgen: the Primary Radiation Damage structure generator
PRDgen is a tool that generates realistic defect structures based on PKA energy and direction.
It utilizes Gaussian Process Regression (GPR) and Convolutional Neural Network (CNN)
Machine A utilize GPR to generate quasi-random defect structure
Machine B utilize CNN to classify whether the defect structure is realistic or not.
PRDgen make iteration between Machine A and B to find reasonable defect structure from certain PKA energy and direction.
The deep description about PRDgen is presented in: JongHyeon Park, Takuji Oda, "Generation of radiation damage structures in bcc-W using machine-learning statistical and image recognition models trained on molecular dynamics", journal of ...


# Machine A
There are four files to operate Machine A.
(1) func_def_gen6.c
- Contains the core algorithms for defect generation and prediction
- Reads probability density files based on energy/direction conditions and samples the number and positions of defects using probability density functions

(2) func_def_gen6.h
- A header file that declares functions used in the func_def_gen6.c file and defines necessary constants

(3) MachineA_numonly.c
- Receives energy, direction, and seed as input, and outputs only the predicted number of defects
- It is utilized to estimate the expected number of defects under given conditions without generating the structure

(4) MachineA_configonly.c
- Receives project name, PKA energy, direction, seed, and defect number, and output is a defect structure
- It is utilized to generate quasi-random structure

## How to use Machine A
1. Compile MachineA_numonly.c and MachineA_configonly.c, respectively. We highly recommend you to name 'def_number.exe' and 'def_config.exe', respectively.
2. You can generate a possible defect number, with PKA energy, direction, and random seed. The result is saved in 'defnum.dat' file.
./def_number.exe 30.0 0.8 0.4 0.2 297297
3. Next, you cen create quasi-random structure, with project name, PKA energy, direction, random seed, and defect number.
./def_config.exe output 30.0 0.8 0.4 0.2 972972 50 > output.dat

# Machine B
- This Python script implements a convolutional neural network (CNN) using Keras to classify 3D defect projection data into two categories

# PRDgen
PRDgen code connect Machine A and B to create realistic defect structure.


## How to use PRDgen
1. You need to first compile a Machine A. (making def_number.exe and def_config.exe)
2. And also, you need two folders: 'alldat-files' and 'limit-files'
3. Then, run the code with desired PKA energy and direction, like:
  python PRDgen.py 30.0 0.5 -0.3 0.9
4. wait a moment, and check the calculation result, and 'primary_damage_structure.dat' file.


# Required files to run PRDgen
All necessary files are shared by Google Drive : https://drive.google.com/drive/folders/14lEF71SVgWBGjOkYrd1lxf828RhxAkVv?usp=sharing
/alldat-files : probability density files to make quasi-random structure by Machine A
/limit-files : highest/lowest values of defect number and defect position at certain PKA energy.
/Final_machine_211109_3.h5 : trained CNN model (result of Machine B)


