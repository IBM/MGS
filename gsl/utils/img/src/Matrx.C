// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

// This file contains matrix classes and functions.
#include <stdio.h>
#include "Matrx.h"
#include "ImgUtil.h"
static Matrx mo; Vectr vo;
Matrx::Matrx(double e00,double e11,double e22)
{
   short i;
   for (i=1;i<8;i++) e[0][i]=0;
   e[0][0]=e00;
   e[1][1]=e11;
   e[2][2]=e22;
}


Matrx::Matrx(double* val)
{
   short i;
   for (i=0;i<9;i++) e[0][i]=val[i];
}


Vectr::Vectr(double e0, double e1, double e2)
{
   e[0] = e0;
   e[1] = e1;
   e[2] = e2;
}


Vectr::Vectr(double e0, double e1)
{
   e[0] = e0;
   e[1] = e1;
   e[2] = 1-e0-e1;
}


Vectr::Vectr(double *val)
{
   for(int i = 0; i<3; i++) e[i] = val[i];
}


Matrx& operator*(Matrx m, Matrx n)
{
   //  Matrx mo;
   short i,j,k;
   double sum;
   for (i=0;i<3;i++)
   for (j=0;j<3;j++) {
      sum = 0;
      for (k=0;k<3;k++)
         sum += m.e[i][k]*n.e[k][j];
      mo.e[i][j] = sum;
   }
   return mo;
}


void MatrxPrt(Matrx m,char* mn,unsigned short lin)
{
   short i;
   if (lin==0) {
      printf("%s\n",mn);
      for (i=0;i<9;i+=3)
         printf("%10f %10f %10f \n",m.e[0][i],m.e[0][i+1],m.e[0][i+2]);
   }
   else {
      printf("%s\nmatrix=",mn);
      printf("%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
         m.e[0][0],m.e[0][1],m.e[0][2],
         m.e[0][3],m.e[0][4],m.e[0][5],
         m.e[0][6],m.e[0][7],m.e[0][8]);
   }
}


void MatrxPrt(Matrx m,char* mn,char* line)
{
   sprintf(line,"%s"
      "%f,%f,%f,%f,%f,%f,%f,%f,%f\r\n",
      mn,
      m.e[0][0],m.e[0][1],m.e[0][2],
      m.e[0][3],m.e[0][4],m.e[0][5],
      m.e[0][6],m.e[0][7],m.e[0][8]);
}


Matrx& operator ~(Matrx m)
{
   //  Matrx mo;
   double det;
   det =  m.e[0][0] * (m.e[1][1]*m.e[2][2] - m.e[1][2]*m.e[2][1])
      -m.e[0][1] * (m.e[1][0]*m.e[2][2] - m.e[1][2]*m.e[2][0])
      +m.e[0][2] * (m.e[1][0]*m.e[2][1] - m.e[1][1]*m.e[2][0]);
   if(det==0) error("Matrix has determinant of 0.");

   mo.e[0][0] = (m.e[1][1]*m.e[2][2] - m.e[1][2]*m.e[2][1]) / det;
   mo.e[1][0] =-(m.e[1][0]*m.e[2][2] - m.e[1][2]*m.e[2][0]) / det;
   mo.e[2][0] = (m.e[1][0]*m.e[2][1] - m.e[1][1]*m.e[2][0]) / det;

   mo.e[0][1] =-(m.e[0][1]*m.e[2][2] - m.e[0][2]*m.e[2][1]) / det;
   mo.e[1][1] = (m.e[0][0]*m.e[2][2] - m.e[0][2]*m.e[2][0]) / det;
   mo.e[2][1] =-(m.e[0][0]*m.e[2][1] - m.e[0][1]*m.e[2][0]) / det;

   mo.e[0][2] = (m.e[0][1]*m.e[1][2] - m.e[0][2]*m.e[1][1]) / det;
   mo.e[1][2] =-(m.e[0][0]*m.e[1][2] - m.e[0][2]*m.e[1][0]) / det;
   mo.e[2][2] = (m.e[0][0]*m.e[1][1] - m.e[0][1]*m.e[1][0]) / det;
   return mo;
}


Matrx& operator *(double s,Matrx m)
{
   //  Matrx mo;
   short i;
   for (i=0;i<9;i++) mo.e[0][i] = s*m.e[0][i];
   return mo;
}


void MatrxChrom(Matrx &mx, double* chrm)
{
   //computes the md matrix from chromaticities
   int i,j;
   double c;
   Matrx m;
   mx.e[0][0] = chrm[0];
   mx.e[1][0] = chrm[1];
   mx.e[2][0] = 1-chrm[0]-chrm[1];
   mx.e[0][1] = chrm[2];
   mx.e[1][1] = chrm[3];
   mx.e[2][1] = 1-chrm[2]-chrm[3];
   mx.e[0][2] = chrm[4];
   mx.e[1][2] = chrm[5];
   mx.e[2][2] = 1-chrm[4]-chrm[5];
   // get inverse of mx
   m = ~mx;
   // use wp to get the columns sizes for md and fix mat
   for (j=0;j<3;j++) {
      c = 3*(m.e[j][0]*chrm[6]+m.e[j][1]*chrm[7]+m.e[j][2]*(1-chrm[6]-chrm[7]));
      for (i=0;i<3;i++)
         mx.e[i][j] *= c;
   }
}


void ChromMatrx(double* chrm, Matrx &mx)
{
   double c1,c2,c3;
   c1 = mx.e[0][0]+mx.e[1][0]+mx.e[2][0];
   c2 = mx.e[0][1]+mx.e[1][1]+mx.e[2][1];
   c3 = mx.e[0][2]+mx.e[1][2]+mx.e[2][2];
   chrm[0]           =  mx.e[0][0]/c1;
   chrm[1]           =  mx.e[1][0]/c1;
   chrm[2]           =  mx.e[0][1]/c2;
   chrm[3]           =  mx.e[1][1]/c2;
   chrm[4]           =  mx.e[0][2]/c3;
   chrm[5]           =  mx.e[1][2]/c3;
   Vectr vw, v(1,1,1);
   vw = mx*v;
   vw = (1/(vw.e[0]+vw.e[1]+vw.e[2]))*vw;
   chrm[6] = vw.e[0];
   chrm[7] = vw.e[1];
}


Vectr& operator *(Matrx m, Vectr v)
{
   //  Vectr vo;
   vo.e[0] = m.e[0][0]*v.e[0] + m.e[0][1]*v.e[1] + m.e[0][2]*v.e[2];
   vo.e[1] = m.e[1][0]*v.e[0] + m.e[1][1]*v.e[1] + m.e[1][2]*v.e[2];
   vo.e[2] = m.e[2][0]*v.e[0] + m.e[2][1]*v.e[1] + m.e[2][2]*v.e[2];
   return vo;
}


Vectr& operator *(double s, Vectr v)
{
   //  Vectr vo;
   short i;
   for (i=0;i<3;i++) vo.e[i] = s*v.e[i];
   return vo;
}


Vectr& operator /(Vectr d, Vectr v)
{
   //  Vectr vo;
   short i;
   for (i=0;i<3;i++) vo.e[i] = d.e[i]/v.e[i];
   return vo;
}


Vectr& operator +(Vectr d, Vectr v)
{
   //  Vectr vo;
   short i;
   for (i=0;i<3;i++) vo.e[i] = d.e[i]+v.e[i];
   return vo;
}


Vectr& operator -(Vectr d, Vectr v)
{
   //  Vectr vo;
   short i;
   for (i=0;i<3;i++) vo.e[i] = d.e[i]-v.e[i];
   return vo;
}


// This function converts xyz to XYZ
Vectr& XYZ(Vectr xxyz,double xY)
{
   //  Vectr vo;
   double sum;
   sum = xY/xxyz.e[1];
   vo.e[1] = xY;
   vo.e[0] = sum*xxyz.e[0];
   vo.e[2] = sum*xxyz.e[2];
   return vo;
}


// This function converts XYZ to xyz
Vectr& xyz(Vectr xXYZ)
{
   double sum;
   sum = 1/(xXYZ.e[0]+xXYZ.e[1]+xXYZ.e[2]);
   return sum*xXYZ;
}


void VectrPrt(Vectr v,char* vn,char* line)
{
   sprintf(line,"%s"
      "%f,%f,%f\r\n",
      vn,
      v.e[0],v.e[1],v.e[2]);
}
