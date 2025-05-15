#include <stdio.h>
#include <stdlib.h>

#include "func_def_gen6.h"

int main(int argc, char *argv[])
{
        int i;
        double energy;
        double direction[3];
        int RS;
        char m_PROJ[500];
        int defnum;
	char fname1[300];
	FILE *fr,*fw;

        if(argc!=8)     {  printf("error: 3 inputs are needed, (1) project name,  (2) energy (double, in keV), (3)(4)(5) direction, (6) random seed (integer >=1) (7) defect number (integer)\n");  exit(1);  }
        sprintf(m_PROJ,"%s",argv[1]);
        energy=atof(argv[2]);
        direction[0] = atof(argv[3]);
        direction[1] = atof(argv[4]);
        direction[2] = atof(argv[5]);
        RS=atoi(argv[6]);
        defnum=atoi(argv[7]);

        func_def_gen2(m_PROJ, energy, direction, RS, defnum);

	return 0;
}

