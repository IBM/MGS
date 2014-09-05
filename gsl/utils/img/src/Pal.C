// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include <stdio.h>
//#include <io.h>
#include <fcntl.h>
#include "Pal.h"
#include "Ini.h"

// class constructor
Pal::Pal()
{
   unsigned short tem[4] = {7,8,4,32};
   Init(tem);
}


Pal::Pal(unsigned short tem[4])
{
   Init(tem);
}


void Pal::Init(unsigned short rgbk[4])
{
   //set up default palette used by OS/2.1
   int i,j,k,n;
   memcpy(rgbkx,rgbk,8);
   n = rgbk[0]*rgbk[1]*rgbk[2];
   if (n+rgbk[3] > 256) error ("Palette too large: r*g*b+k>256.\n");
   for (i=0; i<3; i++) {
      n /= rgbk[i];
      for (j=0,k=0; j<rgbk[i]; j++,k+=n) rgboff[i][j] = k;
   }
   kl = k = rgbk[0]*rgbk[1]*rgbk[2];
   for (j=0; j<rgbk[3]; j++,k++) rgboff[3][j] = k;
}


/* This function gets the current palette from disp.ini,
   converts it to 768 rgb values and gets the max values and deltas
   which determine how the palette is addressed.
   The program makes a palette object and returns its pointer.
*/
Pal* getpal()
{
   // get palette and diffusion data from device.ini file
   IniFile a;
   char name[40], *temptr;
   a.IniOpenRd("disp.ini");
   temptr = a.IniFindApp("current");
   if (temptr==NULL)
      error("Could not find 'current' application in disp.ini.\n");
   temptr = a.IniFindKey("image_disp");
   if (temptr==NULL)
      error("Could not find 'image_disp' key under [current] in disp.ini.\n");
   a.IniGetStr(name);
   temptr = a.IniFindApp(name);
   if (temptr==NULL)
      error("Could not find [%s] application in disp.ini.\n",name);

   //get palette
   temptr = a.IniFindKey("rgbw_num");
   if (temptr==NULL) error("rgbw_num field missing in [%s]\n",name);
   int i,j,k,n,dnum;
   short rgbw[4];
   n = a.IniGetVal(rgbw,4);
   //  printf( "rgbw=%d,%d,%d,%d\n", rgbw[0], rgbw[1], rgbw[2], rgbw[3] );
   if (n<4 || rgbw[0]*rgbw[1]*rgbw[2]+rgbw[3]>256)
      error("Invalid palette in [%s]\n",name);
   Pal *p = new Pal((unsigned short*)rgbw);
   double black[3];
   temptr = a.IniFindKey("K_black");
   if (temptr==NULL)
      memset(black,0,sizeof black);
   else
      a.IniGetVal(black,3);
   double lvl[256];
   double bot,top;
   char color[4][15]= { {
         "red_levels"
      },
      {"green_levels"},
      {"blue_levels"},
      {"gray_levels"}
   };
   double ymax[4];
   for (i=0; i<4; i++) {
      temptr = a.IniFindKey(color[i]);
      if (temptr==NULL) error("Application %s is missing '%s'.\n",color[i]);
      n = a.IniGetVal(&lvl[0],rgbw[i]<<1);
      //    printf( "IniGetVal(%s) = %d\n", color[i], n );
      //    for( int z=0 ; z < n ; ++z ) printf( " %g", lvl[z] );
      //    printf( "\n" );
      if (lvl[n-2]>255) error("Max index, %d, > 255.\n",(int)lvl[n-2]);
      if (n>20 && i<3)
         error("Number of entries for color: %d > 10\n",n/2);
      else if(n>200 && i==3)
         error("Number of entries for gray: %d > 100\n",n/2);
      ymax[i] = lvl[n-1];
      /* p->rgboff[3][10] contains offset addresses into palette for rgb values
         p->koff[100] ditto for gray values
         p->pall[768] contains values in palette = indeces (even lvl values)
         p->rgbcal[3][10] contains normalized values, 0-4095, for rgb measurements
           (odd lvl values)
         p->kcal[100] ditto for gray values
      */
      // normalize levels and save
      bot = black[2];
      top = lvl[n-1]-bot;
      n = rgbw[i]-(n>>1);        //number of zero levels not measured
                                 // Trunc
      dnum = (int) (4095.*(lvl[1]-bot)/top+.5);
      //    printf( "For i=%d, dnum=%g, n=%d, top=%g\n", i, dnum, n, top );
      if (i<3) {
         for (j=0; j<n; j++) {
            p->rgbcal[i][j] = dnum*j*j/(n*n);
            // Trunc
            p->pall[3*p->rgboff[i][j]+i] = (unsigned char) (j*lvl[0]/n);
         }
         for (j=n; j<rgbw[i]; j++) {
            // Trunc
            p->rgbcal[i][j] = (unsigned short) (4095.*(lvl[((j-n)<<1)+1]-bot)/top+.5);
            // Trunc
            p->pall[3*p->rgboff[i][j]+i] = (unsigned char) (lvl[(j-n)<<1]);
         }
      }
      else {
         for (j=0; j<n; j++) {
            p->kcal[j] = dnum*j*j/(n*n);
            p->pall[3*p->koff[j]] =
               p->pall[3*p->koff[j]+1] =
            // Trunc
               p->pall[3*p->koff[j]+2] = (unsigned char) (j*lvl[0]/n);
         }
         for (j=n; j<rgbw[i]; j++) {
            // Trunc
            p->kcal[j] = (unsigned short) (4095.*(lvl[((j-n)<<1)+1]-bot)/top+.5);
            p->pall[3*p->koff[j]] =
               p->pall[3*p->koff[j]+1] =
            // Trunc
               p->pall[3*p->koff[j]+2] = (unsigned short ) (lvl[(j-n)<<1]);
         }
      }
   }
   // fill in all orthogonal palette entries from axis already filled in
   int ix, jx, kx;
   for (i=0; i<rgbw[0]; i++) {
      ix = 3*p->rgboff[0][i];
      for (j=0; j<rgbw[1]; j++) {
         jx = 3*p->rgboff[1][j];
         for (k=0; k<rgbw[2]; k++) {
            kx = 3*p->rgboff[2][k];
            p->pall[ix+jx+kx]   = p->pall[ix];
            p->pall[ix+jx+kx+1] = p->pall[jx+1];
            p->pall[ix+jx+kx+2] = p->pall[kx+2];
         }
      }
   }
   // scale and store matrix mdi and black point
   double white[3];
   Vectr b(black[0],black[1]), w,W,B;
   if( black[2] == 0.0 ) {
      B.e[0] = B.e[1] = B.e[2] = 0.0;
   } else B = XYZ(b,black[2]);
   temptr = a.IniFindKey("rgbw_chromas");
   if (temptr==NULL) error("Could not find 'rgbw_chromas' key.\n");
   n = a.IniGetVal(lvl,8);
   if (n<8)
      error("Less than 8 values in rgbw_chromas field.\n");
   temptr = a.IniFindKey("W_white");
   //  printf( "Computing white...\n" );
   if (temptr==NULL) {           //compute white from RGB Y's
      // add 3 Y's, subtract 2 black Y's, then subtract B from White XYZ
      Vectr wx(lvl[6],lvl[7]);
      W = XYZ(wx,ymax[0]+ymax[1]+ymax[2]-black[2]-black[2]) - B;
   }
   else {
      a.IniGetVal(white,3);
      Vectr wx(white[0],white[1]);
      W = XYZ(wx,white[2])-B;
   }
   //  printf( "Still going...\n" );
   for (i=0; i<3; i++) {
      Vectr x(lvl[i<<1],lvl[(i<<1)+1]),X;
      X = XYZ(x,ymax[i])-B;
      x = xyz(X);
      lvl[i<<1] = x.e[0];
      lvl[(i<<1)+1] = x.e[1];
   }
   //  printf ("Got it\n" );
   w = xyz(W);
   lvl[6] = w.e[0];
   lvl[7] = w.e[1];
   Matrx md;
   MatrxChrom(md,lvl);
   p->mdi = ~md;
   b.e[2] = 4095.*B.e[1]/W.e[1];
   for (i=0;i<3;i++)
      p->black[i] = b.e[i];
   return p;
}
