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

# Machine B



Supporting files to run PRDgen
/alldat-files : probability density files to make quasi-random structure by Machine A
/limit-files : highest/lowest values of defect number and defect position at certain PKA energy.

https://drive.google.com/drive/folders/14lEF71SVgWBGjOkYrd1lxf828RhxAkVv?usp=sharing
