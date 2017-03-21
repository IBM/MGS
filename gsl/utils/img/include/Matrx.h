// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef MATRX_H
#define MATRX_H
#include "Copyright.h"
// This file contains 3x3 matrix classes and functions.

class Matrx
{
   public:
      double e[3][9];
      //constructor
      Matrx(double e11,double e22,double e33);
      Matrx(double* val);
      Matrx(){};
};
Matrx& operator *(Matrx m, Matrx n);
Matrx& operator ~(Matrx m);
Matrx& operator *(double s, Matrx m);
void MatrxChrom(Matrx &mo, double* chrm);
void ChromMatrx(double* chrm, Matrx &mx);
void MatrxPrt(Matrx m,char* mn,unsigned short lin=0);
void MatrxPrt(Matrx m,char* mn,char* line);
class Vectr
{
   public:
      double e[3];
      //constructor
      Vectr(double e0, double e1, double e2);
      Vectr(double e0, double e1);
      Vectr(double *val);
      Vectr(){};
};
Vectr& operator *(Matrx m, Vectr v);
Vectr& operator *(double s, Vectr v);
Vectr& operator /(Vectr d, Vectr v);
Vectr& operator +(Vectr d, Vectr v);
Vectr& operator -(Vectr d, Vectr v);
Vectr& XYZ(Vectr xxyz, double xY);
Vectr& xyz(Vectr xXYZ);
void VectrPrt(Vectr v,char* vn,char* line);
#endif
