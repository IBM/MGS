// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
// pal.h: palette class
#ifndef PAL_H
#include "Copyright.h"
#define PAL_H
#include "Img.h"
#include "Matrx.h"

Pal* getpal();

class Pal
{
   // palette parameters
   unsigned char pall[768];      // rgb values
   unsigned short rgbkx[4];      //max number of red, green, blue, black colors
   unsigned short rgboff[34][32]; //offset location of colors, one must be absolute
   unsigned short koff[100];
   unsigned short kl;            //location of gray values
   unsigned short rgbcal[3][10]; //calibration values
   unsigned short kcal[100];
   friend class TiffImage;
   friend class BMPImage;
   friend class PCXImage;
   friend void diffinit(unsigned short, Pal*, unsigned short*, unsigned short* );
   friend Pal* getpal();
   public:
      Matrx mdi;                 //display matrix
      double black[3];           // xyY black bias
      void Init(unsigned short rgbk[4]);
      // constructor
      Pal(unsigned short rgbk[4]);
      Pal();
};
#endif
