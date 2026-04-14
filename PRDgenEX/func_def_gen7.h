// ver03: 2020/10/21, the error to have defects outside the range is correctd.
// ver04: 2020/10/21, GPR models are implemented (as model03)
// ver05: 2020/10/22, OpenMP implemented
// func_def_gen4: 2024/03/27, random seed change
//
//#define _FLAG_PRINT1    // print data file for OVITO visualization (DPxDPxDP supercell)
//#define _FLAG_ANAL1     // print projected data for machine learning
//#define _FLAG_STDOUT1 // print out basic information
//#define _FLAG_GSL     // if random seed is generated with gsl, please active this and build executable with GSL library

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _FLAG_GSL
    #include <gsl/gsl_rng.h>
#endif

#define _FLAG_OMP
#ifdef _FLAG_OMP
   #include <omp.h>
#endif

////////////////////
//#define _FLAG_PRINT1    // print data file for OVITO visualization (DPxDPxDP supercell)
//#define _FLAG_ANAL1     // print projected data (xls) for machine learning
#define _FLAG_PRINT2  // print (don't save) projected data and OVITO visualization file
#define _FLAG_PRINT3  // print (don't save) data file for OVITO visualization 

#define _LIMIT1 // to add a base probability if the probability density is 0 even if the conditions is within the lower/upper limits.
#define PROB0_NUM 0.002         // 2/10 of the cutoff value
#define PROB0_POS 0.0002        // 2/10 of the cutoff value



#define NF 1    // the number of random structures to be created

#define OFILE_DEFNUM "defnum.dat"	// the file in which the defect number will be printed.


#define INFO_BASE "alldat-files"
#define INFO_FILE "alldat-files/info-summary.txt"
#define LIMIT_SIAx_down "limit-files/limit_SIAx_low.dat"
#define LIMIT_SIAx_up "limit-files/limit_SIAx_high.dat"
#define LIMIT_SIAy_down "limit-files/limit_SIAy_low.dat"
#define LIMIT_SIAy_up "limit-files/limit_SIAy_high.dat"
#define LIMIT_SIAz_down "limit-files/limit_SIAz_low.dat"
#define LIMIT_SIAz_up "limit-files/limit_SIAz_high.dat"
#define LIMIT_VACx_down "limit-files/limit_VACx_low.dat"
#define LIMIT_VACx_up "limit-files/limit_VACx_high.dat"
#define LIMIT_VACy_down "limit-files/limit_VACy_low.dat"
#define LIMIT_VACy_up "limit-files/limit_VACy_high.dat"
#define LIMIT_VACz_down "limit-files/limit_VACz_low.dat"
#define LIMIT_VACz_up "limit-files/limit_VACz_high.dat"
#define LIMIT_num_down "limit-files/limit_number_low.dat"
#define LIMIT_num_up "limit-files/limit_number_high.dat"

#define NINFO 111       // the number of info file.

#define ENE_MAX  32.0   // to normalize the pka energy

#define NDEF 100
#define NPOS 125
#define NENE 320

#define POS0 0.0
#define POSD (1.0/124)
#define ENE0 0.1
#define ENED 0.1


#define SA NPOS
#define SB NPOS
#define SC NPOS
#define SM NPOS // should be the maximum among SA/SB/SC
#define DP NPOS
#define NREF0 2


#define MNA 1500
#define MNE 100

void func_inv_3matrix(double base_mat[][3], double inv_mat[][3]);
int  func_write_gin(char fname1[], int *ne, int *na, double lat2[][3], double pos1[][MNA][3], int sort1[][MNA], char symb1[][10], int Nf);
int func_read_lammpsdata(char fname1[], int *ne, int *na, double lat2[][3], double pos1[][MNA][3], int sort1[][MNA]);
int func_read_lammpsdata_charge(char fname1[], int *ne, int *na, double lat2[][3], double pos1[][MNA][3], int sort1[][MNA], double charge1[][MNA], double vel1[][MNA][3], double mass1[]);
int func_read_lammpsdata_atomic(int flag1, char fname1[], int *ne, int *na, double lat2[][3], double pos1[][MNA][3], int sort1[][MNA], double vel1[][MNA][3], double mass1[]);
double func_dist2(double pos1[], double pos2[]);
double func_dist2_pbc(double pos1[], double pos2[], double lat[][3], double minlat2);
int func_check_consistency(int ne1, int na1, int sort1[][MNA], double lat1[][3], int ne2, int na2, int sort2[][MNA], double lat2[][3]);
int func_convert_to_frac(int ne1, int na1, const int sc[], const int minmax1[][2], double lat2[][3],  double pos2cart[][MNA][3], double pos2frac[][MNA][3]);

double func_rand1();    // return a random fraction in the range of [0,1)

// defect generation model
int func_def_position_model05(int ndef, double pdf_possia[][NPOS], double pdf_posvac[][NPOS], int occ_sia[][SB][SC][NREF0], int occ_vac[][SB][SC][NREF0], int limit_sia[][2], int limit_vac[][2]);
int func_def_position_model06(double pene, int ndef, double pdf_possia[][NPOS], double pdf_posvac[][NPOS], int occ_sia[][SB][SC][NREF0], int occ_vac[][SB][SC][NREF0], int limit_sia[][2], int limit_vac[][2]);

int func_def_number_model05(double pdf_defnum[], int limit_num[]);



int func_limit_lower_SIAx(double ene);
int func_limit_upper_SIAx(double ene);
int func_limit_lower_SIAy(double ene);
int func_limit_upper_SIAy(double ene);
int func_limit_lower_SIAz(double ene);
int func_limit_upper_SIAz(double ene);
int func_limit_lower_VACx(double ene);
int func_limit_upper_VACx(double ene);
int func_limit_lower_VACy(double ene);
int func_limit_upper_VACy(double ene);
int func_limit_lower_VACz(double ene);
int func_limit_upper_VACz(double ene);
int func_limit_lower_num(double ene);
int func_limit_upper_num(double ene);

// random number genneration setting for gsl
#ifdef _FLAG_GSL
  gsl_rng * gslr;
#endif

//void func_def_gen2(char *PROJ, double  pene, double dir[], int defnum, int RS);
void func_def_gen2(char *PROJ, double  energy, double direction[3], int RS, int defnum);
int func_def_num2(double  energy, double direction[3], int RS);

double func_cluster_prob_sia(int i);
double func_cluster_prob_vac(int i);
double func_neig_prob_sia(int i);
double func_neig_prob_vac(int i);

int func_sia_cluster_max(double ene1);
double func_pdf_sia_cluster(double ene1, int nc0);
int func_vac_cluster_max(double ene1);
double func_pdf_vac_cluster(double ene1, int nc0);
int func_check_neighbor(int ndef, int ipos_sia[][4], int a0, int b0, int c0, int s0, double rcut1f);