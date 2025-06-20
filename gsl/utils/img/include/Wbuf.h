// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef WBUF_H
#define WBUF_H
#include "Copyright.h"

class wbuf
{
   int id;
   unsigned char* buf, *buf2;
   char endstr[8];
   unsigned short *ascii;
   int i, size, outsize, dfltflag;
   int compr, finish;

   public:
      //constructor
      wbuf(int idx, int sizex=50000, int comprx=0, const char* endstrx="");

      //destructor
      ~wbuf(void);

      //functions
      void wput(void* ptr,int len);
};
#endif
