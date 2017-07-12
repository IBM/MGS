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

// util.h: utility functions
#ifndef UTIL_H
#include "Copyright.h"
#define UTIL_H

char* strconcat(char * str1, const char* str2 = "", const char* str3 = "",
const char* str4 = "", const char* str5 = "",
const char* str6 = "", const char* str7 = "",
const char* str8 = "", const char* str9 = "");

template <class T> T max(T (x), T (y)) {return ((x)>(y)) ? (x):(y);}
template <class T> T min(T (x), T (y)) {return ((x)<(y)) ? (x):(y);}

void error(const char* msg);
void error(const char* msg, const char* nme);
void error(const char* msg, int val);
void MakeGamma(unsigned short* tbl,double *lvl,int num,unsigned short xmax,
unsigned short,short r,int norm=1);
void GammaTbl(unsigned short* tbl,double *lvl,unsigned short xmax,
unsigned short ymax,short r);
void MakeDither(unsigned short (*dthtbl)[4],unsigned short *rtbl,unsigned short sl);
void decitbl(int w, int h,
double &ws, double &hs,
int* &xio, int* &yio,
int distort=0, int frame=1);
void tblinv(unsigned short* intbl, unsigned short ni,
unsigned short* invtbl, unsigned short no);
#endif
