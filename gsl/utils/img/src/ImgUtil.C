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

/*
12/7/92  created.
This file contains various utility functions used by many programs.
*/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "ImgUtil.h"

#define MIN(a,b) (((a) < (b)) ? (a) : (b))

//concatenates std::strings, str1 must have enough allocated space for result
char* strconcat(char * str1, const char* str2, const char* str3,
const char* str4, const char* str5,
const char* str6, const char* str7,
const char* str8, const char* str9)
{
   return strcat(strcat(strcat(strcat(strcat(strcat(strcat(strcat(
      str1,str2),str3),str4),str5),str6),str7),str8),str9);
}


//handles errors
void error(const char* msg)
{
  printf("%s", msg);
  exit(1);
}


void error(const char* msg, const char* nme)
{
  printf("%s : %s", msg, nme);
  exit(1);
}


void error(const char* msg,int val)
{
  printf("%s : %d", msg, val);
  exit(1);
}


//creates lookup table from points using interpolation in log-log space.
//(can also create inverse lookup table with r=1. Normally r=0.)
//          y = g(ax),  a = xmax/lvl[num-2]
//                      b = ymax/lvl[num-1]
//          table values are scaled: b*y
//          lvl = x[0],y[0],x[1],y[1],... (num values total)
//interpolate in log by vs log ax,     (log y[1] - log y[0])
//                                 m = ----------------------
//                                     (log x[1] - log x[0])
// deltay = m * deltax
// y1-y0  = m * (x1-x0)
//     y1 = m * (x1-x0) +y0
//   log by1 = m*(log ax1 -log ax0) +log by0   because linear in log-log space
//  b*g(ax + delta) = exp(m*( log(ax+delta) - log ax) + log b + log g(ax) )
void MakeGamma(unsigned short* tbl,double *lvl,int num,unsigned short xmax,
unsigned short ymax,short r,int norm)
{
   short i=0,k=0,k2=0,next;
   double x[2],y[2], a,b,m, lyk,axk,laxk,lbyk;
   if (norm==1 && lvl[0]==0) {   //normalize values by subtracting first point
      for (i=3;i<num;i+=2)
         lvl[i] -= lvl[1];
      lvl[1] = 0;
   }
   // set up scale factors a and b
   a = xmax/lvl[num-2+r];
   b = ymax/lvl[num-1-r];

   //check data for zeros
   for (i=0;i<num;i++)
      if (lvl[i]==0) lvl[i] = .000001;
   //starting point
   x[k] = lvl[k2++ +r];
   y[k] = lvl[k2++ -r];
   tbl[0] = 0;
   i = 1;
   k = 1-k;
   while(k2<num) {
      // calculate slope
      x[k] = lvl[k2++ +r];
      y[k] = lvl[k2++ -r];
      k = 1-k;
      lyk = log(y[k]);
      axk = a*x[k];
      laxk = log(axk);
      m = (log(y[1-k]) - lyk) / (log(a*x[1-k]) - laxk);
      lbyk = lyk + log(b);
                                 // Trunc
      next = (short) (a*x[1-k]+.5);
      float conversion;
      while(i<next) {            // interpolate all values to next point
	 conversion = i;
         tbl[i] = (unsigned short) (exp(lbyk + m*(log(conversion)-laxk))+.5);
         i++;
      }
   }
   tbl[i] = (unsigned short) (b*y[1-k]);
}


//creates lookup table from gamma and offset values.
//(can also create inverse lookup table with r=1. Normally r=0.)
//          y = b*g(x),     x = 0,1,...,xmax
//                          b = ymax/g(xmax)
//          g(x) = (x-x0)**gamma
//                          x0 = xmax*lvl[1]
// inverse: x = x0+(y/b)**(1./gamma)
// or       x = x0+(xmax-x0)*y**(1./gamma)
//
//  for r=1: (xmax and ymax interchanged
//          y = y0+(x/b)**(1./gamma)
// or       y = y0+(ymax-y0)*(x/xmax)**(1./gamma)
// where    y0 = ymax*lvl[1]
void GammaTbl(unsigned short* tbl,double *lvl,unsigned short xmax,
unsigned short ymax,short r)
{
   short i=0,k=0;
   double b, x0, y0, gamma;
   // lvl[0] is gamma, lvl[1] is offset (ratio of max)
   if(r==0) {
      x0 = xmax*lvl[1];
      b = ymax/pow(xmax-x0,lvl[0]);
      gamma = lvl[0];
      k = (unsigned short) x0;
      for (i=0;i<k;i++)
         tbl[i] = 0;
      for (;i<=xmax;i++)
         tbl[i] = (unsigned short) (b*pow((double)i-x0,gamma) + .5);
   }
   else {
      y0 = ymax*lvl[1];
      gamma = 1./lvl[0];
      b = (double)(ymax-y0)/pow(xmax,gamma);
      for (i=0;i<=xmax;i++)
         tbl[i] = (unsigned short) (y0+b*pow((double)i,gamma) + .5);
   }
}


void MakeDither(unsigned short (*dthtbl)[4],unsigned short *rtbl,unsigned short sl)
{
   int i,k,m;
   unsigned short ditm[4] = {    // this sets dither pattern
      0,3,1,2
   };
   if (sl>2)
   for (k=0;k<4096;k++) {
      m = rtbl[k]&0x0003;
      for(i=0;i<m;i++) dthtbl[k][ditm[i]] = ((rtbl[k]&~0x0003)+4) << (sl-2);
      for(;   i<4;i++) dthtbl[k][ditm[i]] = ((rtbl[k]&~0x0003) ) << (sl-2);
   }
   else
   for (k=0;k<4096;k++) {
      m = rtbl[k]&0x0003;
      for(i=0;i<m;i++) dthtbl[k][ditm[i]] = ((rtbl[k]&~0x0003)+4) >> (2-sl);
      for(;   i<4;i++) dthtbl[k][ditm[i]] = (rtbl[k]&~0x0003)   >> (2-sl);
   }
}


/*
   This function allocates and creates horizontal and vertical decimation tables
   based on supplied input w and h, and output frame/scale factors wo and ho.
   The resulting width and height will be placed in wo and ho.
*/
void decitbl(int w, int h,
double &ws, double &hs,
int* &xio, int* &yio,
int distort, int frame)
{
   int xo,yo,wo,ho;
   double fw,fh;
   if (frame) {                  //indicates output frame size rather than scale factor
      if (distort == 0) {        // magnification is lesser of wo/w or ho/h
         // THis is not portable from MSVC++; changed, Ravi Rao, 8/28/02
         // fw = __min(ws/w,hs/h);
         fw = MIN(ws/w, hs/h);
         wo = (int) (w*fw+.5);
         ho = (int) (h*fw+.5);
         fw = 1./fw;
         fh = fw;
      }
      else {
         fw = w/ws;
         fh = h/hs;
         wo = (int) (ws+.5);
         ho = (int) (hs+.5);
      }
   }
   else {                        //scale factor
      if (hs == 0) hs=ws;
      fw = 1./ws;
      fh = 1./hs;
      wo = (int) (w/fw+.5);
      ho = (int) (h/fh+.5);
   }
   // calculate xio and yio
   xio=new int[wo+ho];
   if (xio==NULL) error("Could not allocate space for decimation table.\n");
   yio = &xio[0]+wo;
   double f;
   for(xo=0,f=0.00001;xo<wo;xo++,f+=fw)
      xio[xo] = (int) f;
   for(yo=0,f=0.00001;yo<ho;yo++,f+=fh)
      yio[yo] = (int) f;
   ws = wo;
   hs = ho;
}


/*
This function takes an input table, intbl, and an allocated invert table,
invtbl, and creates it. The size of the input and inverse tables are given.
*/
void tblinv(unsigned short* intbl, unsigned short ni,
unsigned short* invtbl, unsigned short no)
{
   unsigned short i,isv,k;
   //initialize invtbl to zero
   memset(invtbl,0,no*sizeof(unsigned short));
   //two step process - first fill in change points
   isv = 0;
   for (i=1;i<ni;i++) {
      k = 1;
      while (intbl[i] == intbl[isv] && i<ni-1) {
         i++;
         k = 0;
      }
      invtbl[(intbl[i]+intbl[isv]+k)>>1] = (i+isv+k)>>1;
      isv = i;
   }
   //then fill in rest of table as needed
   for (i=1;i<no;i++)
      if (invtbl[i] == 0)
         invtbl[i] = invtbl[i-1];
}
