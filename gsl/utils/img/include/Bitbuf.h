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

/*
  This class allows reading and writing bits to a buffer.
#include "Copyright.h"
  id is negative for reading, positive for writing.
*/
#ifndef BITBUF_H
#define BITBUF_H

#include "Wbuf.h"

class Bitbuf
{
   unsigned char* buf;
   unsigned char* byteptr;
   int id;
   int bufsize;
   int bitoffset;
   int bytecount;
   int readflag;
   int resetflag;

   wbuf *wa;
   void writebuf(void);
   void reset(void);

   public:
      //constructor
      Bitbuf(int xid, int xbufsize, int comprx, const char* enstrx);

      //destructor
      ~Bitbuf(void);

                                 //sequentially writes data to buffer
      int putbits(int data, int bitcount);
      int getbits(int bitcount); //sequentially reads data from buffer
      void setreset(void) { resetflag = 1; }
};
#endif
