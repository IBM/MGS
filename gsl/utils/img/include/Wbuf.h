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
