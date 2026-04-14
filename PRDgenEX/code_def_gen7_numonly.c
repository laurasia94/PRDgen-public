#include <stdio.h>
#include <stdlib.h>

#include "func_def_gen7.h"

int main(int argc, char *argv[])
{
        double energy;
        double direction[3];
        int RS;

        if(argc!=6)     {  printf("error: 5 inputs are needed, (1) energy (double, in keV), (2)(3)(4) direction, (5) random seed (integer >=1)\n");  exit(1);  }
        energy=atof(argv[1]);
        direction[0] = atof(argv[2]);
        direction[1] = atof(argv[3]);
        direction[2] = atof(argv[4]);
        RS=atoi(argv[5]);

	int defnum;
        defnum=func_def_num2(energy, direction, RS); 

        printf("%d\n", defnum);

	return 0;
}

