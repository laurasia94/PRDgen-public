// func_def_gen4: 2024/03/27, random seed change

#include "func_def_gen6.h"

void func_def_gen2(char *PROJ, double pene, double dir[], int RS, int defnum)
{

	int r,i,j,k,m;
	int a,b,c;
	int a0,b0,c0;
	int ne1,na1,ne2,na2;
	int nn1,nn2;
	int jud1;
	int m0,m1,m2;
	static int occ_sia[SA][SB][SC][NREF0];
        static int occ_vac[SA][SB][SC][NREF0];
	int (*fp_limit_upper_sia[3])(double)={func_limit_upper_SIAx, func_limit_upper_SIAy, func_limit_upper_SIAz};
        int (*fp_limit_lower_sia[3])(double)={func_limit_lower_SIAx, func_limit_lower_SIAy, func_limit_lower_SIAz};
        int (*fp_limit_upper_vac[3])(double)={func_limit_upper_VACx, func_limit_upper_VACy, func_limit_upper_VACz};
        int (*fp_limit_lower_vac[3])(double)={func_limit_lower_VACx, func_limit_lower_VACy, func_limit_lower_VACz};
	int limit_sia[3][2]={{0,NPOS-1},{0,NPOS-1},{0,NPOS-1}};
	int limit_vac[3][2]={{0,NPOS-1},{0,NPOS-1},{0,NPOS-1}};
	int limit_num[2]={0,NDEF-1};
	double pdf_defnum[NDEF];
	double pdf_posvac[3][NPOS];
	double pdf_possia[3][NPOS];

	static int proj_xy[DP][DP][2],proj_yz[DP][DP][2],proj_xz[DP][DP][2],proj_xyz[DP][DP][DP][2];	// [][][0]:SIA, [][][1]:vac
	int item1[300];
	double dtem1[300];
	char fname1[500];

	double refpos0[NREF0][3]={
		{0.0, 0.0, 0.0},
		{0.5, 0.5, 0.5}
	};
///	double dir[3];
	int ndef[2];	// ndef[0]:nsia, ndef[1]:nvac
	//int NF=1;
	FILE *fw,*fr;
///	char PROJ[500];

	double info_data1[NINFO][2];
	double info_data2[NINFO][3];
	char info_name[NINFO][300];
	double dmin;
	int idmin;
	char cline1[100000];
	int iene;


	dtem1[0]=0.0;
	for(i=0;i<3;i++)	dtem1[0]+=pow(dir[i],2.0);
	dtem1[0]=sqrt(dtem1[0]);
	for(i=0;i<3;i++)	dir[i]/=dtem1[0];
	if(dir[0]<dir[1] || dir[1]<dir[2] || dir[2]<0)	{  printf("error in input, direction informtion\n");  exit(1);  }
	if(pene>32.0)	{  printf("error in input, pene\n");  exit(1);  }
	double zz,phi;
	zz=dir[2];
	phi=dir[1]/sqrt(1-zz*zz);
	phi=asin(phi);

//// revised 2024/03/27, for more random number, srand with (unsigned)time(NULL) and large prime number 1131617
	srand((unsigned)time(NULL)+RS*1131617);

// reading the summary file of .dat files information
	if( (fr=fopen(INFO_FILE,"r"))==NULL )	{  printf("error in file reading\n");  exit(1); }
	for(i=0;i<NINFO;i++)
	{
		fscanf(fr,"%d",&j);
		if(i!=j)	{  printf("error 1\n");  exit(1);  }
		for(j=0;j<2;j++)	fscanf(fr,"%lf",&info_data1[i][j]);
		fscanf(fr,"%s",info_name[i]);
		for(j=0;j<3;j++)	fscanf(fr,"%lf",&info_data2[i][j]);
	}


	////////////
	//  finding the nearest reference point
	///////////  
	dmin=func_dist2(dir, info_data2[0]);	// square distance
	idmin=0;
	for(i=1;i<NINFO;i++)
	{
		dtem1[0]=func_dist2(dir, info_data2[i]);
		if(dtem1[0]<dmin)
		{
			dmin=dtem1[0];
			idmin=i;
		}
	}
#ifdef _FLAG_STDOUT1
	printf("energy: %lf keV\n",pene);
	printf("direction: %lf %lf %lf\n",dir[0],dir[1],dir[2]);
	printf("nearest reference point: %lf %lf %lf\n",info_data2[idmin][0],info_data2[idmin][1],info_data2[idmin][2]);
#endif


// find and read the nearest .dat file
	if(pene<ENE0)	iene=0;
	else
	{
		dtem1[0]=(pene-ENE0)/ENED;
		iene=(int)(0.5+dtem1[0]);
	}
	if(iene<0 || iene>=NENE)	{  printf("error 1\n");  exit(1);  }

	sprintf(fname1,"%s/%s",INFO_BASE,info_name[idmin]);
	if( (fr=fopen(fname1,"r"))==NULL )	{  printf("error in file open1:%s\n",info_name[idmin]);  exit(1);  }
	for(i=0;i<iene;i++)
	{
		fgets(cline1,100000,fr);
		if( (strncmp(cline1,"Energy",6))!=0 )	{  printf("error in Energy word at %d\n",i);  exit(1);  }
		for(j=0;j<7;j++)	fgets(cline1,100000,fr);
	}
	fscanf(fr,"%s",cline1);
	if( (strncmp(cline1,"Energy",6))!=0 )   {  printf("error in Energy word2\n");  exit(1);  }
	fscanf(fr,"%lf",&dtem1[0]);
	if(fabs(dtem1[0]-(ENE0+iene*ENED)/ENE_MAX)>0.0001)	{  printf("error in energy\n");  exit(1);  };

	fscanf(fr,"%s",cline1);
      	if( (strncmp(cline1,"Number",6))!=0 )   {  printf("error in Number word2\n");  exit(1);  }
	for(i=0;i<NDEF;i++)	fscanf(fr,"%lf",&pdf_defnum[i]);
      	fscanf(fr,"%s",cline1);
        
       	if( (strncmp(cline1,"SIAx",4))!=0 )   {  printf("error in Number SIAx\n");  exit(1);  }
       	for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_possia[0][i]);

       	fscanf(fr,"%s",cline1);
      	if( (strncmp(cline1,"SIAy",4))!=0 )   {  printf("error in Number SIAy\n");  exit(1);  }
       	for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_possia[1][i]);

       	fscanf(fr,"%s",cline1);
      	if( (strncmp(cline1,"SIAz",4))!=0 )   {  printf("error in Number SIAz\n");  exit(1);  }
      	for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_possia[2][i]);

      	fscanf(fr,"%s",cline1);
       	if( (strncmp(cline1,"VACx",4))!=0 )   {  printf("error in Number VACx\n");  exit(1);  }
      	for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_posvac[0][i]);

      	fscanf(fr,"%s",cline1);
      	if( (strncmp(cline1,"VACy",4))!=0 )   {  printf("error in Number VACy\n");  exit(1);  }
      	for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_posvac[1][i]);

    	fscanf(fr,"%s",cline1);
     	if( (strncmp(cline1,"VACz",4))!=0 )   {  printf("error in Number VACz\n");  exit(1);  }
      	for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_posvac[2][i]);
	fclose(fr);


// set the limit
#ifdef _LIMIT1
        for(i=0;i<3;i++)
        {
                limit_sia[i][0]=fp_limit_lower_sia[i](pene);
                limit_sia[i][1]=fp_limit_upper_sia[i](pene);
                limit_vac[i][0]=fp_limit_lower_vac[i](pene);
                limit_vac[i][1]=fp_limit_upper_vac[i](pene);
        }
        limit_num[0]=func_limit_lower_num(pene);
        limit_num[1]=func_limit_upper_num(pene);
#ifdef _FLAG_STDOUT1
        printf("number-limit(lower,upper): %d %d\n",limit_num[0],limit_num[1]);
        printf("sia-lower-position-limit: %d %d %d\n",limit_sia[0][0],limit_sia[1][0],limit_sia[2][0]);
        printf("sia-upper-position-limit: %d %d %d\n",limit_sia[0][1],limit_sia[1][1],limit_sia[2][1]);
        printf("vac-lower-position-limit: %d %d %d\n",limit_vac[0][0],limit_vac[1][0],limit_vac[2][0]);
        printf("vac-upper-position-limit: %d %d %d\n",limit_vac[0][1],limit_vac[1][1],limit_vac[2][1]);
	printf("prob0-to-be-added(number,position): %lf %lf\n",PROB0_NUM,PROB0_POS);
#endif

	for(i=0;i<NDEF;i++)
	{
		if(pdf_defnum[i]==0.00 && i>=limit_num[0] && i<=limit_num[1])	pdf_defnum[i]+=PROB0_NUM;
		
		if( (i<limit_num[0] || i>limit_num[1]) && fabs(pdf_defnum[i])>0.001)	
		{  
			printf("error-pdf1 at %d:%e %d %d\n",i,pdf_defnum[i],limit_num[0],limit_num[1]);  
			for(j=0;j<NDEF;j++)	printf("%d\t%lf\n",j,pdf_defnum[j]);
			exit(1);  
		}
	}
	for(i=0;i<3;i++)
	{
		for(j=0;j<NPOS;j++)
		{
			if(pdf_possia[i][j]==0.00 && j>=limit_sia[i][0] && j<=limit_sia[i][1])	pdf_possia[i][j]+=PROB0_POS;
                        if(pdf_posvac[i][j]==0.00 && j>=limit_vac[i][0] && j<=limit_vac[i][1])  pdf_posvac[i][j]+=PROB0_POS;

               		if( (j<limit_sia[i][0] || j>limit_sia[i][1]) && fabs(pdf_possia[i][j])>0.0001)  {  printf("error-pdf2a\n");  exit(1);  }
			if( (j<limit_vac[i][0] || j>limit_vac[i][1]) && fabs(pdf_posvac[i][j])>0.0001)  {  printf("error-pdf2b\n");  exit(1);  }
		}
	}

#endif


// generating structures
        for(r=0;r<NF;r++)
        {
#ifdef _FLAG_STDOUT1
		printf("%d out of %d\n",r,NF);		
#endif
	
		for(a=0;a<SA;a++)	for(b=0;b<SB;b++)	for(c=0;c<SC;c++)	for(i=0;i<NREF0;i++)	occ_sia[a][b][c][i]=0;
		for(a=0;a<SA;a++)       for(b=0;b<SB;b++)       for(c=0;c<SC;c++)       for(i=0;i<NREF0;i++)    occ_vac[a][b][c][i]=0;
        	for(i=0;i<DP;i++)
                for(j=0;j<DP;j++)
              	for(k=0;k<2;k++)
               	{
                   	proj_xy[i][j][k]=0;
                       	proj_yz[i][j][k]=0;
                      	proj_xz[i][j][k]=0;
                        proj_xyz[i][j][m][k]=0;
             	}


		//ndef[0]=func_def_number_model05(pdf_defnum, limit_num);
		ndef[0]=defnum;
		ndef[1]=ndef[0];
	
		func_def_position_model05(ndef[0], pdf_possia, pdf_posvac, occ_sia, occ_vac, limit_sia, limit_vac);

#ifdef _FLAG_PRINT1
		sprintf(fname1,"data.dummy-defect_%s_%d",PROJ,r);
		if( (fw=fopen(fname1,"w"))==NULL )	{  printf("error in dummy-data file open\n"); exit(1);  }
		fprintf(fw,"LAMMPS data file only defects SIA %d vac %d\n",ndef[0],ndef[0]);
		fprintf(fw,"\n");
		fprintf(fw,"%d atoms\n",ndef[0]+ndef[1]);
		fprintf(fw,"2 atom types\n");		
		fprintf(fw,"\n");
		fprintf(fw,"0.0 %.2lf xlo xhi\n",1.0*SA);
                fprintf(fw,"0.0 %.2lf ylo yhi\n",1.0*SB);
                fprintf(fw,"0.0 %.2lf zlo zhi\n",1.0*SC);
		fprintf(fw,"\n");
		fprintf(fw,"Masses\n");
		fprintf(fw,"\n");
		fprintf(fw,"1 1\n");
		fprintf(fw,"2 2\n");
		fprintf(fw,"\n");
		fprintf(fw,"Atoms\n");
		fprintf(fw,"\n");

		m0=0;
		m1=0;
		m2=0;
                for(a=0;a<SA;a++)
                for(b=0;b<SB;b++)
                for(c=0;c<SC;c++)
                for(i=0;i<NREF0;i++)
                {
			// sia printing out
			if(occ_sia[a][b][c][i]==1)
			{
				m0++;
				m1++;
				fprintf(fw,"%d 1  %.3lf %.3lf %.3lf\n",m0,a+refpos0[i][0],b+refpos0[i][1],c+refpos0[i][2]);
			}

			// vacancy printing out
                        if(occ_vac[a][b][c][i]==1)
			{
				m0++;
				m2++;
				fprintf(fw,"%d 2  %.3lf %.3lf %.3lf\n",m0,a+refpos0[i][0],b+refpos0[i][1],c+refpos0[i][2]);
                        }
                }
		fclose(fw);

		if(m1!=ndef[0])		{  printf("error: mismatch in nsia\n");  exit(1);  }
                if(m2!=ndef[1])        	{  printf("error: mismatch in nvac\n");  exit(1);  }
#endif

#ifdef _FLAG_ANAL1
                for(a=0;a<SA;a++)
                for(b=0;b<SB;b++)
                for(c=0;c<SC;c++)
                for(i=0;i<NREF0;i++)
                {
			// sia projection
                        if(occ_sia[a][b][c][i]==1)
                        {
				proj_xy[a][b][0]++;
                                proj_yz[b][c][0]++;
                                proj_xz[a][c][0]++;
                                proj_xyz[a][b][c][0]++;
                        }

			// vacancy projection
                        if(occ_vac[a][b][c][i]==1)
                        {
                                proj_xy[a][b][1]++;
                                proj_yz[b][c][1]++;
                                proj_xz[a][c][1]++;
                                proj_xyz[a][b][c][1]++;
                        }
                }
                
		sprintf(fname1,"%s_%d_xy.xls",PROJ,r);
                if( (fw=fopen(fname1,"w"))==NULL )      {  printf("error in data file open\n"); exit(1);  }
		for(i=0;i<DP;i++)
		{
			for(j=0;j<DP;j++)
			{
				a=i;
               	         	b=j;
				fprintf(fw,"%d\t%d\t%d\t%d\n",i,j,proj_xy[a][b][0],proj_xy[a][b][1]);
			}
		}
		fclose(fw);

                sprintf(fname1,"%s_%d_yz.xls",PROJ,r);
                if( (fw=fopen(fname1,"w"))==NULL )      {  printf("error in data file open\n"); exit(1);  }
                for(i=0;i<DP;i++)
                {
                        for(j=0;j<DP;j++)
                        {
                                b=i;
                                c=j;
                              	fprintf(fw,"%d\t%d\t%d\t%d\n",i,j,proj_yz[b][c][0],proj_yz[b][c][1]);
                        }
                }
                fclose(fw);

                sprintf(fname1,"%s_%d_xz.xls",PROJ,r);
                if( (fw=fopen(fname1,"w"))==NULL )      {  printf("error in data file open\n"); exit(1);  }
                for(i=0;i<DP;i++)
                {
                        for(j=0;j<DP;j++)
                        {
                                a=i;
                                c=j;
                              	fprintf(fw,"%d\t%d\t%d\t%d\n",i,j,proj_xz[a][c][0],proj_xz[a][c][1]);
                        }
                }
                fclose(fw);
#endif

#ifdef _FLAG_PRINT2
                for(a=0;a<SA;a++)
                for(b=0;b<SB;b++)
                for(c=0;c<SC;c++)
                for(i=0;i<NREF0;i++)
                {
			// sia projection
                        if(occ_sia[a][b][c][i]==1)
                        {
				proj_xy[a][b][0]++;
                                proj_yz[b][c][0]++;
                                proj_xz[a][c][0]++;
                                proj_xyz[a][b][c][0]++;
                        }

			// vacancy projection
                        if(occ_vac[a][b][c][i]==1)
                        {
                                proj_xy[a][b][1]++;
                                proj_yz[b][c][1]++;
                                proj_xz[a][c][1]++;
                                proj_xyz[a][b][c][1]++;
                        }
                }
                
                // Output results to the console instead of files
                //XY
                for(i=0;i<DP;i++)
                {
                        for(j=0; j<DP; j++)
                        {
                                a = i;
                                b = j;
                                printf("%d\t%d\t%d\t%d\n",i,j,proj_xy[a][b][0],proj_xy[a][b][1]);
                        }
                }
                printf("FLAG_cut\n");

                //YZ
                for(i=0;i<DP;i++)
                {
                        for(j=0; j<DP; j++)
                        {
                                b = i;
                                c = j;
                                printf("%d\t%d\t%d\t%d\n",i,j,proj_yz[b][c][0],proj_yz[b][c][1]);
                        }
                }
                printf("FLAG_cut\n");

                //ZX
                for(i=0;i<DP;i++)
                {
                        for(j=0; j<DP; j++)
                        {
                                a = i;
                                c = j;
                                printf("%d\t%d\t%d\t%d\n",i,j,proj_xz[a][c][0],proj_xz[a][c][1]);
                        }
                }
                printf("FLAG_cut\n");
#endif

#ifdef _FLAG_PRINT3
		printf("LAMMPS data file only defects SIA %d vac %d\n",ndef[0],ndef[0]);
		printf("\n");
		printf("%d atoms\n",ndef[0]+ndef[1]);
		printf("2 atom types\n");		
		printf("\n");
		printf("0.0 %.2lf xlo xhi\n",1.0*SA);
                printf("0.0 %.2lf ylo yhi\n",1.0*SB);
                printf("0.0 %.2lf zlo zhi\n",1.0*SC);
		printf("\n");
		printf("Masses\n");
		printf("\n");
		printf("1 1\n");
		printf("2 2\n");
		printf("\n");
		printf("Atoms\n");
		printf("\n");

		m0=0;
		m1=0;
		m2=0;
                for(a=0;a<SA;a++)
                for(b=0;b<SB;b++)
                for(c=0;c<SC;c++)
                for(i=0;i<NREF0;i++)
                {
			// sia printing out
			if(occ_sia[a][b][c][i]==1)
			{
				m0++;
				m1++;
				printf("%d 1  %.3lf %.3lf %.3lf\n",m0,a+refpos0[i][0],b+refpos0[i][1],c+refpos0[i][2]);
			}

			// vacancy printing out
                        if(occ_vac[a][b][c][i]==1)
			{
				m0++;
				m2++;
				printf("%d 2  %.3lf %.3lf %.3lf\n",m0,a+refpos0[i][0],b+refpos0[i][1],c+refpos0[i][2]);
                        }
                }

		if(m1!=ndef[0])		{  printf("error: mismatch in nsia\n");  exit(1);  }
                if(m2!=ndef[1])        	{  printf("error: mismatch in nvac\n");  exit(1);  }
                printf("FLAG_visual\n");
#endif
	}	
}
/// revised 2024/03/27, random seed RSS is needed for defect number decision
int func_def_num2(double pene, double dir[], int RSS)
{

        int r,i,j,k;
        int a,b,c;
        int a0,b0,c0;
        int ne1,na1,ne2,na2;
        int nn1,nn2;
        int jud1;
        int m0,m1,m2;
        static int occ_sia[SA][SB][SC][NREF0];
        static int occ_vac[SA][SB][SC][NREF0];
        int (*fp_limit_upper_sia[3])(double)={func_limit_upper_SIAx, func_limit_upper_SIAy, func_limit_upper_SIAz};
        int (*fp_limit_lower_sia[3])(double)={func_limit_lower_SIAx, func_limit_lower_SIAy, func_limit_lower_SIAz};
        int (*fp_limit_upper_vac[3])(double)={func_limit_upper_VACx, func_limit_upper_VACy, func_limit_upper_VACz};
        int (*fp_limit_lower_vac[3])(double)={func_limit_lower_VACx, func_limit_lower_VACy, func_limit_lower_VACz};
        int limit_sia[3][2]={{0,NPOS-1},{0,NPOS-1},{0,NPOS-1}};
        int limit_vac[3][2]={{0,NPOS-1},{0,NPOS-1},{0,NPOS-1}};
        int limit_num[2]={0,NDEF-1};
        double pdf_defnum[NDEF];
        double pdf_posvac[3][NPOS];
        double pdf_possia[3][NPOS];

        static int proj_xy[DP][DP][2],proj_yz[DP][DP][2],proj_xz[DP][DP][2];    // [][][0]:SIA, [][][1]:vac
        int item1[300];
        double dtem1[300];
        char fname1[500];

        double refpos0[NREF0][3]={
                {0.0, 0.0, 0.0},
                {0.5, 0.5, 0.5}
        };
///     double dir[3];
        int ndef[2];    // ndef[0]:nsia, ndef[1]:nvac
        //int NF=1;
        FILE *fw,*fr;
///     char PROJ[500];

        double info_data1[NINFO][2];
        double info_data2[NINFO][3];
        char info_name[NINFO][300];
        double dmin;
        int idmin;
        char cline1[100000];
        int iene;


        dtem1[0]=0.0;
        for(i=0;i<3;i++)        dtem1[0]+=pow(dir[i],2.0);
        dtem1[0]=sqrt(dtem1[0]);
        for(i=0;i<3;i++)        dir[i]/=dtem1[0];
        if(dir[0]<dir[1] || dir[1]<dir[2] || dir[2]<0)  {  printf("error in input, direction informtion\n");  exit(1);  }
        if(pene>32.0)       {  printf("error in input, pene\n");  exit(1);  }
        double zz,phi;
        zz=dir[2];
        phi=dir[1]/sqrt(1-zz*zz);
        phi=asin(phi);

//// random number generation by computer clock
//// revised 2024/03/27, for more random number, srand with (unsigned)time(NULL) and large prime number 1131617

        srand((unsigned)time(NULL)+RSS*1131617);
/////////////////


// reading the summary file of .dat files information
        if( (fr=fopen(INFO_FILE,"r"))==NULL )   {  printf("error in file reading\n");  exit(1); }
        for(i=0;i<NINFO;i++)
        {
                fscanf(fr,"%d",&j);
                if(i!=j)        {  printf("error 1\n");  exit(1);  }
                for(j=0;j<2;j++)        fscanf(fr,"%lf",&info_data1[i][j]);
                fscanf(fr,"%s",info_name[i]);
                for(j=0;j<3;j++)        fscanf(fr,"%lf",&info_data2[i][j]);
        }


        ////////////
        //  finding the nearest reference point
        ///////////  
        dmin=func_dist2(dir, info_data2[0]);    // square distance
        idmin=0;
        for(i=1;i<NINFO;i++)
        {
                dtem1[0]=func_dist2(dir, info_data2[i]);
                if(dtem1[0]<dmin)
                {
                        dmin=dtem1[0];
                        idmin=i;
                }
        }
#ifdef _FLAG_STDOUT1
        printf("energy: %lf keV\n",pene);
        printf("direction: %lf %lf %lf\n",dir[0],dir[1],dir[2]);
        printf("nearest reference point: %lf %lf %lf\n",info_data2[idmin][0],info_data2[idmin][1],info_data2[idmin][2]);
#endif


// find and read the nearest .dat file
        if(pene<ENE0)   iene=0;
        else
        {
                dtem1[0]=(pene-ENE0)/ENED;
                iene=(int)(0.5+dtem1[0]);
        }
        if(iene<0 || iene>=NENE)        {  printf("error 1\n");  exit(1);  }

        sprintf(fname1,"%s/%s",INFO_BASE,info_name[idmin]);
        if( (fr=fopen(fname1,"r"))==NULL )      {  printf("error in file open1:%s\n",info_name[idmin]);  exit(1);  }
        for(i=0;i<iene;i++)
        {
                fgets(cline1,100000,fr);
                if( (strncmp(cline1,"Energy",6))!=0 )   {  printf("error in Energy word at %d\n",i);  exit(1);  }
                for(j=0;j<7;j++)        fgets(cline1,100000,fr);
        }
        fscanf(fr,"%s",cline1);

        if( (strncmp(cline1,"Energy",6))!=0 )   {  printf("error in Energy word2\n");  exit(1);  }
        fscanf(fr,"%lf",&dtem1[0]);
        if(fabs(dtem1[0]-(ENE0+iene*ENED)/ENE_MAX)>0.0001)      {  printf("error in energy\n");  exit(1);  };

        fscanf(fr,"%s",cline1);
        if( (strncmp(cline1,"Number",6))!=0 )   {  printf("error in Number word2\n");  exit(1);  }
        for(i=0;i<NDEF;i++)     fscanf(fr,"%lf",&pdf_defnum[i]);

        fscanf(fr,"%s",cline1);
        if( (strncmp(cline1,"SIAx",4))!=0 )   {  printf("error in Number SIAx\n");  exit(1);  }
        for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_possia[0][i]);

        fscanf(fr,"%s",cline1);
        if( (strncmp(cline1,"SIAy",4))!=0 )   {  printf("error in Number SIAy\n");  exit(1);  }
        for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_possia[1][i]);

        fscanf(fr,"%s",cline1);
        if( (strncmp(cline1,"SIAz",4))!=0 )   {  printf("error in Number SIAz\n");  exit(1);  }
        for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_possia[2][i]);

        fscanf(fr,"%s",cline1);
        if( (strncmp(cline1,"VACx",4))!=0 )   {  printf("error in Number VACx\n");  exit(1);  }
        for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_posvac[0][i]);

        fscanf(fr,"%s",cline1);
        if( (strncmp(cline1,"VACy",4))!=0 )   {  printf("error in Number VACy\n");  exit(1);  }
        for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_posvac[1][i]);

        fscanf(fr,"%s",cline1);
        if( (strncmp(cline1,"VACz",4))!=0 )   {  printf("error in Number VACz\n");  exit(1);  }
        for(i=0;i<NPOS;i++)     fscanf(fr,"%lf",&pdf_posvac[2][i]);
        fclose(fr);


// set the limit
#ifdef _LIMIT1
        for(i=0;i<3;i++)
        {
                limit_sia[i][0]=fp_limit_lower_sia[i](pene);
                limit_sia[i][1]=fp_limit_upper_sia[i](pene);
                limit_vac[i][0]=fp_limit_lower_vac[i](pene);
                limit_vac[i][1]=fp_limit_upper_vac[i](pene);
        }
        limit_num[0]=func_limit_lower_num(pene);
        limit_num[1]=func_limit_upper_num(pene);
#ifdef _FLAG_STDOUT1
        printf("number-limit(lower,upper): %d %d\n",limit_num[0],limit_num[1]);
        printf("sia-lower-position-limit: %d %d %d\n",limit_sia[0][0],limit_sia[1][0],limit_sia[2][0]);
        printf("sia-upper-position-limit: %d %d %d\n",limit_sia[0][1],limit_sia[1][1],limit_sia[2][1]);
        printf("vac-lower-position-limit: %d %d %d\n",limit_vac[0][0],limit_vac[1][0],limit_vac[2][0]);
        printf("vac-upper-position-limit: %d %d %d\n",limit_vac[0][1],limit_vac[1][1],limit_vac[2][1]);
        printf("prob0-to-be-added(number,position): %lf %lf\n",PROB0_NUM,PROB0_POS);
#endif

        for(i=0;i<NDEF;i++)
        {
                if(pdf_defnum[i]==0.00 && i>=limit_num[0] && i<=limit_num[1])   pdf_defnum[i]+=PROB0_NUM;

                if( (i<limit_num[0] || i>limit_num[1]) && fabs(pdf_defnum[i])>0.001)
                {
                        printf("error-pdf1 at %d:%e %d %d\n",i,pdf_defnum[i],limit_num[0],limit_num[1]);
                        for(j=0;j<NDEF;j++)     printf("%d\t%lf\n",j,pdf_defnum[j]);
                        exit(1);
                }
        }
        for(i=0;i<3;i++)
        {
                for(j=0;j<NPOS;j++)
                {
                        if(pdf_possia[i][j]==0.00 && j>=limit_sia[i][0] && j<=limit_sia[i][1])  pdf_possia[i][j]+=PROB0_POS;
                        if(pdf_posvac[i][j]==0.00 && j>=limit_vac[i][0] && j<=limit_vac[i][1])  pdf_posvac[i][j]+=PROB0_POS;

                        if( (j<limit_sia[i][0] || j>limit_sia[i][1]) && fabs(pdf_possia[i][j])>0.0001)  {  printf("error-pdf2a\n");  exit(1);  }
                        if( (j<limit_vac[i][0] || j>limit_vac[i][1]) && fabs(pdf_posvac[i][j])>0.0001)  {  printf("error-pdf2b\n");  exit(1);  }
                }
        }

#endif

        ndef[0]=func_def_number_model05(pdf_defnum, limit_num);

	/*
        if( (fw=fopen(OFILE_DEFNUM,"w"))==NULL )        {  printf("error in file open\n");  exit(1);  }
        fprintf(fw,"%d\n",ndef[0]);
        fclose(fw);
	*/
	return ndef[0];
}






void func_inv_3matrix(double base_mat[][3], double inv_mat[][3])
{
        double buf;
        double mat[3][3];
        int i,j,k;
        const int n=3;

        for(i=0;i<n;i++)
                for(j=0;j<n;j++)
                {
                        mat[i][j]=base_mat[i][j];
                        if(i==j)
                                inv_mat[i][j]=1.0;
                        else
                                inv_mat[i][j]=0.0;
                }

        for(i=0;i<n;i++)
        {
                buf=1.0/mat[i][i];
                for(j=0;j<n;j++)
                {
                        mat[i][j]*=buf;
                        inv_mat[i][j]*=buf;
                }
                for(j=0;j<n;j++)
                {
                        if(i!=j)
                        {
                                buf=mat[j][i];
                                for(k=0;k<n;k++)
                                {
                                        mat[j][k]-=mat[i][k]*buf;
                                        inv_mat[j][k]-=inv_mat[i][k]*buf;
                                }
                        }
                }
        }
}


int  func_write_gin(char fname1[], int *ne, int *na, double lat2[][3], double pos1[][MNA][3], int sort1[][MNA], char symb1[][10], int Nf)
{
	int i,j,k,n,m,m1,m2;
	double pos1_t[3];
	double lat2_inv[3][3];
	FILE *fw;

        func_inv_3matrix(lat2, lat2_inv);
        printf("lattic constants (and inverse)\n");
        printf("matrix\n");
        for(i=0;i<3;i++)
        {
                for(j=0;j<3;j++)
                        printf("%lf\t",lat2[i][j]);
                printf("\n");
        }
        printf("inv-matrix\n");
        for(i=0;i<3;i++)
        {
                for(j=0;j<3;j++)
                        printf("%lf\t",lat2_inv[i][j]);
                printf("\n");
        }

	
	if( (fw=fopen(fname1,"w"))==NULL )		
	{
		printf("cannot open the file in func_write_gin\n");
		exit(1);
	}
		

        fprintf(fw,"single\n");
        fprintf(fw,"title\n");
        fprintf(fw,"temp\n");
        fprintf(fw,"end\n");
        fprintf(fw,"vectors\n");
        for(i=0;i<3;i++)
                fprintf(fw,"%.8lf  %.8lf  %.8lf\n",lat2[i][0],lat2[i][1],lat2[i][2]);
        fprintf(fw,"frac %d\n",*na);
        for(n=0;n<*na;n++)
        {
                m1=n/MNA;
                m2=n%MNA;

                for(i=0;i<3;i++)
                {
                        pos1_t[i]=0.0;
                        for(j=0;j<3;j++)
                                pos1_t[i]+=(pos1[m1][m2][j]*lat2_inv[j][i]);
                        pos1_t[i]=(pos1_t[i]+100)-(int)(pos1_t[i]+100);
                }
                fprintf(fw,"%s core  %.8lf %.8lf %.8lf\n",symb1[sort1[m1][m2]],
                        pos1_t[0],pos1_t[1],pos1_t[2]);
        }
        fprintf(fw,"space\n");
        fprintf(fw,"1\n");

	fclose(fw);

	return 1;
}


int func_read_lammpsdata(char fname1[], int *ne, int *na, double lat2[][3], double pos1[][MNA][3], int sort1[][MNA])
{
	int i,j,k,n,m;
	int m1,m2;
	int item1[30];
	double dtem1[30];
	char cline1[300],cline2[300],cline3[300];
	const int mna2=MNA*MNA;
	double lat1[9];
	FILE *fr[2];

        if( (fr[0]=fopen(fname1,"r"))==NULL )
        {
                printf("cannot open the file in func_read_lammpsdata\n");
		printf("%s\n",fname1);
                exit(1);
        }

        fgets(cline1,300,fr[0]);
        fscanf(fr[0],"%d",na);
        fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"atoms",5))!=0 )
        {
                printf("error: here should be \"atoms\" but \"%s\"\n",cline1);
                exit(1);
        }
        if( *na>mna2)
        {
                printf("error: too many atoms\n");
                exit(1);
        }

        fscanf(fr[0],"%d",&i);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"atom",4))!=0  || (strncmp(cline2,"types",5))!=0 )
        {
                printf("error: here should be \"atom types\" but \"%s %s\"\n",cline1,cline2);
                exit(1);
        }
        if( i>MNE)
        {
                printf("error: too many atoms\n");
                exit(1);
        }
        else if(i!=*ne)
        {
                printf("error: the number of elements is inconsistent-1: %d %d\n",i,*ne);
                exit(1);
        }
        printf("the number of atoms:\t%d\n",*na);
        printf("the number of elements:\t%d\n",*ne);

        for(i=0;i<MNA;i++)
                for(j=0;j<MNA;j++)
                        sort1[i][j]=-100;

        fscanf(fr[0],"%lf",&lat1[0]);
        fscanf(fr[0],"%lf",&lat1[1]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"xlo",3))!=0 || (strncmp(cline2,"xhi",3))!=0 )
        {
                printf("error: here should be \"xlo\" and \"xhi\"  but \"%s\" and \"%s\"\n",cline1,cline2);
                exit(1);
        }
        fscanf(fr[0],"%lf",&lat1[2]);
        fscanf(fr[0],"%lf",&lat1[3]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"ylo",3))!=0 || (strncmp(cline2,"yhi",3))!=0 )
        {
                printf("error: here should be \"ylo\" and \"yhi\"  but \"%s\" and \"%s\"\n",cline1,cline2);
                exit(1);
        }
        fscanf(fr[0],"%lf",&lat1[4]);
        fscanf(fr[0],"%lf",&lat1[5]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"zlo",3))!=0 || (strncmp(cline2,"zhi",3))!=0 )
        {
                printf("error: here should be \"zlo\" and \"zhi\"  but \"%s\" and \"%s\"\n",cline1,cline2);
                exit(1);
        }
	lat1[6]=0.0;	
	lat1[7]=0.0;
	lat1[8]=0.0;
#ifdef _TILT
        fscanf(fr[0],"%lf",&lat1[6]);
        fscanf(fr[0],"%lf",&lat1[7]);
        fscanf(fr[0],"%lf",&lat1[8]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        fscanf(fr[0],"%s",cline3);
        if( (strncmp(cline1,"xy",2))!=0 || (strncmp(cline2,"xz",2))!=0 || (strncmp(cline3,"yz",2))!=0)
        {
                printf("error: here should be \"xy\", \"xz\" and \"yz\" but \"%s\", \"%s\" and \"%s\"\n",cline1,cline2,cline3);
                exit(1);
        }
#endif

        fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"Masses",6))!=0 )
        {
                printf("error: here should be \"Masses\" but \"%s\"\n",cline1);
                exit(1);
        }
        for(i=0;i<*ne;i++)
        {
                fscanf(fr[0],"%d",&item1[0]);
                fscanf(fr[0],"%lf",&dtem1[0]);
                if(item1[0]!=(i+1))
                {
                        printf("error in Masses:%d-%d\n",item1[0],i+1);
                        exit(1);
                }
        }

        fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"Atoms",5))!=0 )
        {
                printf("error: here should be \"Atoms\" but \"%s\"\n",cline1);
                exit(1);
        }

        for(n=0;n<*na;n++)
        {
                fscanf(fr[0],"%d",&item1[0]);
                item1[0]--;
                if(item1[0]<0 || item1[0]>=*na)
                {
                        printf("error in atom-id\n");
                        exit(1);
                }
                m1=item1[0]/MNA;
                m2=item1[0]%MNA;
                if(sort1[m1][m2]>=0)
                {
                        printf("error: this atom-id emerged more then twice\n");
                        exit(1);
                }
                fscanf(fr[0],"%d",&item1[1]);
                item1[1]--;
                if(item1[1]<0 || item1[1]>=*ne)
                {
                        printf("error in the sort of atom: %d %d\n",item1[1],*ne);
                        exit(1);
                }
                sort1[m1][m2]=item1[1];
                for(i=0;i<3;i++)
                        fscanf(fr[0],"%lf",&pos1[m1][m2][i]);
                fgets(cline1,300,fr[0]);
#ifdef _ATOM_POS_SHIFT
                for(i=0;i<3;i++)        pos1[m1][m2][i]-=lat1[2*i];
#endif
        }

        lat2[0][0]=lat1[1]-lat1[0];
        lat2[0][1]=0.0;
        lat2[0][2]=0.0;
        lat2[1][0]=lat1[6];
        lat2[1][1]=lat1[3]-lat1[2];
        lat2[1][2]=0.0;
        lat2[2][0]=lat1[7];
        lat2[2][1]=lat1[8];
        lat2[2][2]=lat1[5]-lat1[4];

	fclose(fr[0]);

	return 0;
}

int func_read_lammpsdata_charge(char fname1[], int *ne, int *na, double lat2[][3], double pos1[][MNA][3], int sort1[][MNA], double charge1[][MNA], double vel1[][MNA][3], double mass1[])
{
        int i,j,k,n,m;
        int m1,m2;
        int item1[30];
        double dtem1[30];
        char cline1[300],cline2[300],cline3[300];
        const int mna2=MNA*MNA;
        double lat1[9];
        FILE *fr[2];

        if( (fr[0]=fopen(fname1,"r"))==NULL )
        {
                printf("cannot open the file in func_read_lammpsdata\n");
                printf("%s\n",fname1);
                exit(1);
        }

        fgets(cline1,300,fr[0]);
        fscanf(fr[0],"%d",na);
        fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"atoms",5))!=0 )
        {
                printf("error: here should be \"atoms\" but \"%s\"\n",cline1);
                exit(1);
        }
        if( *na>mna2)
        {
                printf("error: too many atoms\n");
                exit(1);
        }

        fscanf(fr[0],"%d",&i);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"atom",4))!=0  || (strncmp(cline2,"types",5))!=0 )
        {
                printf("error: here should be \"atom types\" but \"%s %s\"\n",cline1,cline2);
                exit(1);
        }
        if( i>MNE)
        {
                printf("error: too many atoms\n");
                exit(1);
        }
        else if(i!=*ne)
        {
                printf("error: the number of elements is inconsistent-2 in %s: %d %d\n",fname1,i,*ne);
                exit(1);
        }
        printf("the number of atoms:\t%d\n",*na);
        printf("the number of elements:\t%d\n",*ne);

        for(i=0;i<MNA;i++)
                for(j=0;j<MNA;j++)
                        sort1[i][j]=-100;

        fscanf(fr[0],"%lf",&lat1[0]);
        fscanf(fr[0],"%lf",&lat1[1]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"xlo",3))!=0 || (strncmp(cline2,"xhi",3))!=0 )
        {
                printf("error: here should be \"xlo\" and \"xhi\"  but \"%s\" and \"%s\"\n",cline1,cline2);
                exit(1);
        }
        fscanf(fr[0],"%lf",&lat1[2]);
        fscanf(fr[0],"%lf",&lat1[3]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"ylo",3))!=0 || (strncmp(cline2,"yhi",3))!=0 )
        {
                printf("error: here should be \"ylo\" and \"yhi\"  but \"%s\" and \"%s\"\n",cline1,cline2);
                exit(1);
        }
        fscanf(fr[0],"%lf",&lat1[4]);
        fscanf(fr[0],"%lf",&lat1[5]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"zlo",3))!=0 || (strncmp(cline2,"zhi",3))!=0 )
        {
                printf("error: here should be \"zlo\" and \"zhi\"  but \"%s\" and \"%s\"\n",cline1,cline2);
                exit(1);
        }
	lat1[6]=0.0;
	lat1[7]=0.0;
	lat1[8]=0.0;
#ifdef _TILT
        fscanf(fr[0],"%lf",&lat1[6]);
        fscanf(fr[0],"%lf",&lat1[7]);
        fscanf(fr[0],"%lf",&lat1[8]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        fscanf(fr[0],"%s",cline3);
        if( (strncmp(cline1,"xy",2))!=0 || (strncmp(cline2,"xz",2))!=0 || (strncmp(cline3,"yz",2))!=0)
        {
                printf("error: here should be \"xy\", \"xz\" and \"yz\" but \"%s\", \"%s\" and \"%s\"\n",cline1,cline2,cline3);
                exit(1);
        }
#endif

        fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"Masses",6))!=0 )
        {
                printf("error: here should be \"Masses\" but \"%s\"\n",cline1);
                exit(1);
        }
        for(i=0;i<*ne;i++)
        {
                fscanf(fr[0],"%d",&item1[0]);
                fscanf(fr[0],"%lf",&mass1[i]);
                if(item1[0]!=(i+1))
                {
                        printf("error in Masses:%d-%d\n",item1[0],i+1);
                        exit(1);
                }
        }

	fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"Atoms",5))!=0 )
        {
                printf("error: here should be \"Atoms\" but \"%s\"\n",cline1);
                exit(1);
        }
	fgets(cline1,300,fr[0]);

        for(n=0;n<*na;n++)
        {
                fscanf(fr[0],"%d",&item1[0]);
                item1[0]--;
                if(item1[0]<0 || item1[0]>=*na)
                {
                        printf("error in atom-id\n");
                        exit(1);
                }
                m1=item1[0]/MNA;
                m2=item1[0]%MNA;
                if(sort1[m1][m2]>=0)
                {
                        printf("error: this atom-id emerged more then twice\n");
                        exit(1);
                }
                fscanf(fr[0],"%d",&item1[1]);
                item1[1]--;
                if(item1[1]<0 || item1[1]>=*ne)
                {
                        printf("error in the sort of atom-2 in %s: %d %d\n",fname1,item1[1],*ne);
                        exit(1);
                }
                sort1[m1][m2]=item1[1];
		fscanf(fr[0],"%lf",&charge1[m1][m2]);
                for(i=0;i<3;i++)
                        fscanf(fr[0],"%lf",&pos1[m1][m2][i]);
                fgets(cline1,300,fr[0]);
#ifdef _ATOM_POS_SHIFT
                for(i=0;i<3;i++)        pos1[m1][m2][i]-=lat1[2*i];
#endif
        }

	fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"Velocities",10))!=0 )
        {
                printf("error: here should be \"Velocities\" but \"%s\"\n",cline1);
                exit(1);
        }
        fgets(cline1,300,fr[0]);

        for(n=0;n<*na;n++)
        {
                fscanf(fr[0],"%d",&item1[0]);
                item1[0]--;
                if(item1[0]<0 || item1[0]>=*na)
                {
                        printf("error in atom-id\n");
                        exit(1);
                }
                m1=item1[0]/MNA;
                m2=item1[0]%MNA;
		for(i=0;i<3;i++)	fscanf(fr[0],"%lf",&vel1[m1][m2][i]);
	}


        lat2[0][0]=lat1[1]-lat1[0];
        lat2[0][1]=0.0;
        lat2[0][2]=0.0;
        lat2[1][0]=lat1[6];
        lat2[1][1]=lat1[3]-lat1[2];
        lat2[1][2]=0.0;
        lat2[2][0]=lat1[7];
        lat2[2][1]=lat1[8];
        lat2[2][2]=lat1[5]-lat1[4];

        fclose(fr[0]);

        return 0;
}

int func_read_lammpsdata_atomic(int flag1, char fname1[], int *ne, int *na, double lat2[][3], double pos1[][MNA][3], int sort1[][MNA], double vel1[][MNA][3], double mass1[])
{
        int i,j,k,n,m;
        int m1,m2;
        int item1[30];
        double dtem1[30];
        char cline1[300],cline2[300],cline3[300];
        const int mna2=MNA*MNA;
        double lat1[9];
        FILE *fr[2];

        if( (fr[0]=fopen(fname1,"r"))==NULL )
        {
                printf("cannot open the file in func_read_lammpsdata\n");
                printf("%s\n",fname1);
                exit(1);
        }

        fgets(cline1,300,fr[0]);
        fscanf(fr[0],"%d",na);
        fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"atoms",5))!=0 )
        {
                printf("error: here should be \"atoms\" but \"%s\"\n",cline1);
                exit(1);
        }
        if( *na>mna2)
        {
                printf("error: too many atoms\n");
                exit(1);
        }

        fscanf(fr[0],"%d",&i);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"atom",4))!=0  || (strncmp(cline2,"types",5))!=0 )
        {
                printf("error: here should be \"atom types\" but \"%s %s\"\n",cline1,cline2);
                exit(1);
        }
        if( i>MNE)
        {
                printf("error: too many atoms\n");
                exit(1);
        }
        else if(i!=*ne)
        {
                printf("error: the number of elements is inconsistent-2 in %s: %d %d\n",fname1,i,*ne);
                exit(1);
        }
        printf("the number of atoms:\t%d\n",*na);
        printf("the number of elements:\t%d\n",*ne);

        for(i=0;i<MNA;i++)
                for(j=0;j<MNA;j++)
                        sort1[i][j]=-100;

        fscanf(fr[0],"%lf",&lat1[0]);
        fscanf(fr[0],"%lf",&lat1[1]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"xlo",3))!=0 || (strncmp(cline2,"xhi",3))!=0 )
        {
                printf("error: here should be \"xlo\" and \"xhi\"  but \"%s\" and \"%s\"\n",cline1,cline2);
                exit(1);
        }
        fscanf(fr[0],"%lf",&lat1[2]);
        fscanf(fr[0],"%lf",&lat1[3]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"ylo",3))!=0 || (strncmp(cline2,"yhi",3))!=0 )
        {
                printf("error: here should be \"ylo\" and \"yhi\"  but \"%s\" and \"%s\"\n",cline1,cline2);
                exit(1);
        }
        fscanf(fr[0],"%lf",&lat1[4]);
        fscanf(fr[0],"%lf",&lat1[5]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        if( (strncmp(cline1,"zlo",3))!=0 || (strncmp(cline2,"zhi",3))!=0 )
        {
                printf("error: here should be \"zlo\" and \"zhi\"  but \"%s\" and \"%s\"\n",cline1,cline2);
                exit(1);
        }
        lat1[6]=0.0;
        lat1[7]=0.0;
        lat1[8]=0.0;
#ifdef _TILT
        fscanf(fr[0],"%lf",&lat1[6]);
        fscanf(fr[0],"%lf",&lat1[7]);
        fscanf(fr[0],"%lf",&lat1[8]);
        fscanf(fr[0],"%s",cline1);
        fscanf(fr[0],"%s",cline2);
        fscanf(fr[0],"%s",cline3);
        if( (strncmp(cline1,"xy",2))!=0 || (strncmp(cline2,"xz",2))!=0 || (strncmp(cline3,"yz",2))!=0)
        {
                printf("error: here should be \"xy\", \"xz\" and \"yz\" but \"%s\", \"%s\" and \"%s\"\n",cline1,cline2,cline3);
                exit(1);
        }
#endif

        fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"Masses",6))!=0 )
        {
                printf("error: here should be \"Masses\" but \"%s\"\n",cline1);
                exit(1);
        }
        for(i=0;i<*ne;i++)
        {
                fscanf(fr[0],"%d",&item1[0]);
                fscanf(fr[0],"%lf",&mass1[i]);
                if(item1[0]!=(i+1))
                {
                        printf("error in Masses:%d-%d\n",item1[0],i+1);
                        exit(1);
                }
        }

        fscanf(fr[0],"%s",cline1);
        if( (strncmp(cline1,"Atoms",5))!=0 )
        {
                printf("error: here should be \"Atoms\" but \"%s\"\n",cline1);
                exit(1);
        }
        fgets(cline1,300,fr[0]);

        for(n=0;n<*na;n++)
        {
                fscanf(fr[0],"%d",&item1[0]);
                item1[0]--;
                if(item1[0]<0 || item1[0]>=*na)
                {
                        printf("error in atom-id\n");
                        exit(1);
                }
                m1=item1[0]/MNA;
                m2=item1[0]%MNA;
                if(sort1[m1][m2]>=0)
                {
                        printf("error: this atom-id emerged more then twice\n");
                        exit(1);
                }
                fscanf(fr[0],"%d",&item1[1]);
                item1[1]--;
                if(item1[1]<0 || item1[1]>=*ne)
                {
                        printf("error in the sort of atom-2 in %s: %d %d\n",fname1,item1[1],*ne);
                        exit(1);
                }
                sort1[m1][m2]=item1[1];
//                fscanf(fr[0],"%lf",&charge1[m1][m2]);
                for(i=0;i<3;i++)
                        fscanf(fr[0],"%lf",&pos1[m1][m2][i]);
                fgets(cline1,300,fr[0]);
#ifdef _ATOM_POS_SHIFT
                for(i=0;i<3;i++)        pos1[m1][m2][i]-=lat1[2*i];
#endif
        }

	if(flag1==0)
	{
		for(n=0;n<*na;n++)	
		{
	                m1=n/MNA;
                	m2=n%MNA;
                	for(i=0;i<3;i++) 	vel1[m1][m2][i]=0.0;
		}

	}
	else
	{
        	fscanf(fr[0],"%s",cline1);
        	if( (strncmp(cline1,"Velocities",10))!=0 )
        	{
			printf("%d\n",*na);
                	printf("error: here should be \"Velocities\" but \"%s\"\n",cline1);
			printf("file %s\n",fname1);
			for(i=0;i<10;i++)
			{
				fgets(cline1,300,fr[0]);
				printf("%d\t%s",i,cline1);
			}
                	exit(1);
        	}
        	fgets(cline1,300,fr[0]);

        	for(n=0;n<*na;n++)
        	{
                	fscanf(fr[0],"%d",&item1[0]);
                	item1[0]--;
                	if(item1[0]<0 || item1[0]>=*na)
                	{
                        	printf("error in atom-id\n");
                       	 	exit(1);
                	}
                	m1=item1[0]/MNA;
                	m2=item1[0]%MNA;
                	for(i=0;i<3;i++)        fscanf(fr[0],"%lf",&vel1[m1][m2][i]);
        	}
	}


        lat2[0][0]=lat1[1]-lat1[0];
        lat2[0][1]=0.0;
        lat2[0][2]=0.0;
        lat2[1][0]=lat1[6];
        lat2[1][1]=lat1[3]-lat1[2];
        lat2[1][2]=0.0;
        lat2[2][0]=lat1[7];
        lat2[2][1]=lat1[8];
        lat2[2][2]=lat1[5]-lat1[4];

        fclose(fr[0]);

        return 0;
}




                                                       

double func_dist2(double pos1[], double pos2[])
{
        double val;
        val=pow(pos1[0]-pos2[0],2.0)+pow(pos1[1]-pos2[1],2.0)+pow(pos1[2]-pos2[2],2.0);
        return val;
}

double func_dist2_pbc(double pos1[], double pos2[], double lat[][3], double minlat2)
{
	int i,a,b,c;
        double mind=1000000.0;
	double pos2rev[3];
	double dist2;

	dist2=func_dist2(pos1,pos2);
	if(dist2<minlat2)
	{
		return dist2;
	}
	else
	{
		for(a=-1;a<=1;a++)
		{
			for(b=-1;b<=1;b++)
			{
				for(c=-1;c<=1;c++)
				{
					for(i=0;i<3;i++)	pos2rev[i]=pos2[i]+a*lat[0][i]+b*lat[1][i]+c*lat[2][i];
					dist2=func_dist2(pos1,pos2rev);
					if(dist2<mind)	
					{
						mind=dist2;
					}
				}
			}
		}

		return mind;
	}
//	printf("  %lf\t%lf %lf %lf\n",mind,pos2min[0],pos2min[1],pos2min[2]);
}

int func_check_consistency(int ne1, int na1, int sort1[][MNA], double lat1[][3], int ne2, int na2, int sort2[][MNA], double lat2[][3])
{	
	int i,j;

    	if(ne2!=ne1)
	{
            	printf("error-2: the number of elements\n");
		return 1;
	}

   	if(na2!=na1)
	{
          	printf("error: the number of atoms is strange: na1=%d, na2=%d\n",na1,na2);
		return 1;
	}

  	for(i=0;i<3;i++)
      	{
              	for(j=0;j<3;j++)
               	{
                   	if(fabs(lat1[i][j]-lat2[i][j])>0.0001)
			{
                           	printf("error in constant-volume: please check the volume between reference and samples\n");
                               	printf("%lf : %lf  at %d-%d\n",lat1[i][j],lat2[i][j],i,j);
					
				return 1;
                        }
                }
	}

	return 0;
}


int func_convert_to_frac(int ne1, int na1, const int sc[], const int minmax1[][2], double lat2[][3],  double pos2cart[][MNA][3], double pos2frac[][MNA][3])
{
        int i,j,k;
        int a,b,c;
        int n,n1,n2;
        int jud1;
        int item1[30];
        double lat2_unit[3][3],lat2_unit_inv[3][3];
        double mod_ref_cart[3];
        double dtem1[30];

        for(i=0;i<3;i++)
                for(j=0;j<3;j++)
                        lat2_unit[i][j]=lat2[i][j]/sc[i];
        func_inv_3matrix(lat2_unit,lat2_unit_inv);


         for(n=0;n<na1;n++)
        {
                n1=n/MNA;
                n2=n%MNA;


                for(i=0;i<3;i++)
                {
                        pos2frac[n1][n2][i]=0.0;
                        for(j=0;j<3;j++)
                                pos2frac[n1][n2][i]+=(pos2cart[n1][n2][j]*lat2_unit_inv[j][i]);
                        if(pos2frac[n1][n2][i]<minmax1[i][0] || pos2frac[n1][n2][i]>minmax1[i][1])
                        {
                                printf("error: fractional coordinates are out of range at sample-%d\n",n);
                                printf("%lf\t",pos2frac[n1][n2][i]);
                                return 1;
                        }
                }
	}
	return 0;
}

double func_rand1()	//[0,1)
{
	double val;
#ifdef _FLAG_GSL
	val=gsl_rng_uniform(gslr);
#else
	val=rand()/(1.0+RAND_MAX);
#endif
	return val;
}


int func_def_position_model05(int ndef, double pdf_possia[][NPOS], double pdf_posvac[][NPOS], int occ_sia[][SB][SC][NREF0], int occ_vac[][SB][SC][NREF0], int limit_sia[][2], int limit_vac[][2])
{
	// the probability density functions should be normalized so that that maximum probability is below and near 1.
	// the origin of the functions should correspond to the PKA position.
        int i,j;
        int jud1;
        double dtem1,dtem2;
	int a0,b0,c0;

        // for SIA
        j=0;
        do
        {
                a0=limit_sia[0][0]+(int)((limit_sia[0][1]-limit_sia[0][0]+1)*func_rand1());
                b0=limit_sia[1][0]+(int)((limit_sia[1][1]-limit_sia[1][0]+1)*func_rand1());
                c0=limit_sia[2][0]+(int)((limit_sia[2][1]-limit_sia[2][0]+1)*func_rand1());
                i=(int)(NREF0*func_rand1());
		if(a0<0 || b0<0 || c0<0 || a0>limit_sia[0][1] || b0>limit_sia[1][1] || c0>limit_sia[2][1] || a0>=SA || b0>=SB || c0>=SC)	{  printf("errro2\n");  exit(1);  }

		if(occ_sia[a0][b0][c0][i]==0)
               	{
			dtem1=func_rand1();
			if(dtem1<pdf_possia[0][a0])
			{
                        	dtem1=func_rand1();
                    		if(dtem1<pdf_possia[1][b0])
                        	{
                                	dtem1=func_rand1();
					if(dtem1<pdf_possia[2][c0])
					{
                				occ_sia[a0][b0][c0][i]+=1;
                     				j++;
					}
				}
			}
		}
        }while(j<ndef);


	// for vacancy
        j=0;
        do
        {
                a0=limit_vac[0][0]+(int)((limit_vac[0][1]-limit_vac[0][0]+1)*func_rand1());
                b0=limit_vac[1][0]+(int)((limit_vac[1][1]-limit_vac[1][0]+1)*func_rand1());
                c0=limit_vac[2][0]+(int)((limit_vac[2][1]-limit_vac[2][0]+1)*func_rand1());
                i=(int)(NREF0*func_rand1());
                if(a0<0 || b0<0 || c0<0 || a0>limit_vac[0][1] || b0>limit_vac[1][1] || c0>limit_vac[2][1] || a0>=SA || b0>=SB || c0>=SC)        {  printf("errro2\n");  exit(1);  }


                if(occ_vac[a0][b0][c0][i]==0)
                {
                        dtem1=func_rand1();
                        if(dtem1<pdf_posvac[0][a0])
                        {
                                dtem1=func_rand1();
                                if(dtem1<pdf_posvac[1][b0])
                                {
                                        dtem1=func_rand1();
                                        if(dtem1<pdf_posvac[2][c0])
                                        {
                                                occ_vac[a0][b0][c0][i]+=1;
                                                j++;
                                        }
                                }
                        }
                }
        }while(j<ndef);

}

int func_def_number_model05(double pdf_defnum[], int limit_num[])
{
	int i,j,k;
	int nn;
	int jud1;
	double dtem1,dtem2;


	jud1=0;
	do
	{
		nn=limit_num[0]+(int)((limit_num[1]-limit_num[0]+1)*func_rand1());
		if(nn<limit_num[0] || nn>limit_num[1] || nn>=NDEF)	{  printf("error1\n");  exit(1);  }
		dtem1=func_rand1();
		if(dtem1<pdf_defnum[nn])	jud1++;
	}while(jud1==0);

	return nn;
}



int func_limit_lower_SIAx(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_SIAx_down, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}

int func_limit_upper_SIAx(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_SIAx_up, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}


int func_limit_lower_SIAy(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_SIAy_down, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}
int func_limit_upper_SIAy(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_SIAy_up, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}

int func_limit_lower_SIAz(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_SIAz_down, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}
int func_limit_upper_SIAz(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_SIAz_up, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}

int func_limit_lower_VACx(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_VACx_down, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}
int func_limit_upper_VACx(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_VACx_up, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}


int func_limit_lower_VACy(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_VACy_down, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}
int func_limit_upper_VACy(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_VACy_up, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}

int func_limit_lower_VACz(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_VACz_down, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}
int func_limit_upper_VACz(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_VACz_up, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}

int func_limit_lower_num(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_num_down, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
  {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}
int func_limit_upper_num(double ene)
{
  double eval;
  FILE *fr2;
  int i, lim;
  fr2 = fopen(LIMIT_num_up, "r");
  if(fr2 == NULL)
  {
    printf("error in file reading\n");
    exit(1);
  }
  else
    {
    for(i=0;i<NENE;i++)
    {
      fscanf(fr2, "%lf %d", &eval, &lim);
      if(ene-eval>0)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }
  fclose(fr2);
  return lim;
}

