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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>

using namespace std;
#include "Img.h"

#define PRINT_IN_INFO 1

int ReadImage(char *in_fname, unsigned char **buffer_ptr, int *nrows_ptr, int *ncols_ptr)
{
   int wm;

   unsigned char *in_buffer;
   int  spp;                     // samples per pixel

   // Files
   ImageType in_itype;
   Image *in_img;
   int nrows,ncols,size;
   short *bps;

   // *********************************************************
   // Read input image
   // *********************************************************
   in_itype = GetType(in_fname);
   in_img = Header(in_itype);
   in_img->ReadHeader(in_fname);
   nrows = in_img->height();
   ncols = in_img->width();
   spp=in_img->GetSpp();
   bps=in_img->GetBits();
   wm=in_img->Getnw();
   size = in_img->GetSize();

   *nrows_ptr = nrows;
   *ncols_ptr = ncols;

   // size = w*h?
   if (PRINT_IN_INFO) {
      printf("%s%i\n","Size: ",size);
      printf("%s%i\n","W: ",ncols);
      printf("%s%i\n","H: ",nrows);
      printf("%s%i\n","Spp: ",spp);
      printf("%s%i\n","Bps: ",bps[0]);
      // printf("%s%i\n","Btyes per line (nw): ",wm);
   }
   if (bps[0]!=8 || spp!=1) {
      std::cerr << "Wrong image type. Image must be grayscale (1 sample per pixel, 8 bits per pixel)" << std::endl;
      return -1;
   }

   // OpenLine() reads a single line, OpenLine(h) reads entire img
   in_img->OpenLine(nrows);
   in_buffer = (unsigned char*)in_img->GetLine();

   #ifdef CHECK
   // A test to check if image has been read correctly
   int ii, jj;
   std::cout << "Printing Image" << std::endl;
   for(ii = 0; ii < nrows; ii++) {
      for(jj = 0; jj < ncols; jj++) {
         printf("%d ",  in_buffer[ii*cols+jj]);
      }
      printf("\n");
   }
   #endif

   *buffer_ptr = in_buffer;
   return 1;

}
