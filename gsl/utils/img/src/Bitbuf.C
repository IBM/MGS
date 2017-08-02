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
  This class allows reading and writing bits to a buffer.
  Reading indicated by negative bufsize parameter.
  Encoding codes for comprx when writing:
  0 - none
  1 - 12 bits per component, ASCII85
  2 -  8 bits per component, ASCIIHEX
  3 - 12 bits per component, ASCIIHEX
  4 -  8 bits per component, ASCII85
*/
//#include <io.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "ImgUtil.h"
#include "Bitbuf.h"
#include "Wbuf.h"

//constructor
Bitbuf::Bitbuf(int idx, int xbufsize, int comprx, const char* endstrx)
{
   if (xbufsize > 0) {
      readflag = 0;
      bufsize = xbufsize;
      wa = new wbuf(idx, xbufsize, comprx, endstrx);
      buf = new unsigned char[bufsize+5];
      if (buf == NULL) error("Could not allocate space for bitbuf.\n");
   }
   else {
      readflag = 1;
      bufsize = -xbufsize;
      id = idx;
      buf = new unsigned char[bufsize+5];
      if (buf == NULL) error("Could not allocate space for bitbuf.\n");
   }
   resetflag = 1;
}


//destructor
Bitbuf::~Bitbuf(void)
{
   if (readflag == 0) {
      //write remainder of data
      if (bitoffset != 0) bytecount++;
      // write(id,buf,bytecount);
      wa->wput(buf, bytecount);
      delete wa;
   }
   delete[] buf;
}


// input data is right justified in int
                                 //sequentially writes data to buffer
int Bitbuf::putbits(int data, int bitcount)
{
   int i;
   union
   {
      int word;
      unsigned char byte[4];
   }out;

   if (resetflag != 0) reset();
   if (bitcount > 16) error("Bit count must be 16 or less.\n");
   if (bytecount+3 > bufsize)
      writebuf();
                                 //shift ranges from 1 to 23
   out.word = data<<(24-bitcount-bitoffset);
   //reverse bits for storage, note byte[3] is 0
   byteptr[0] |= out.byte[2];
   byteptr[1] |= out.byte[1];
   byteptr[2] |= out.byte[0];
   bitoffset += bitcount;
   i = bitoffset>>3;
   byteptr += i;
   bytecount += i;
   bitoffset &= 7;
   return bytecount;
}


void Bitbuf::writebuf(void)
{
   // write(id,buf,bytecount);
   wa->wput(buf,bytecount);
   buf[0] = buf[bytecount];
   memset(buf+1, 0, bufsize-1);
   byteptr = buf;
   bytecount = 0;
}


int Bitbuf::getbits(int bitcount)//sequentially reads data from buffer
{
   union
   {
      unsigned int word;
      unsigned char byte[4];
   }out;
   if (resetflag != 0) reset();
   if (bytecount+3 > bufsize) {
      memcpy(buf,&buf[bytecount],bufsize-bytecount);
      bufsize = read(id,&buf[bufsize-bytecount],bufsize);
      bytecount = 0;
   }

   //reverse bits for storage, note out.byte[0] final is not used
   out.byte[3] = buf[bytecount];
   out.byte[2] = buf[bytecount+1];
   out.byte[1] = buf[bytecount+2];
   out.word <<= bitoffset;
   out.word >>= 32-bitcount;
   bitoffset += bitcount;
   bytecount += bitoffset>>3;
   bitoffset &= 7;
   return (int)out.word;
}


void Bitbuf::reset(void)
{
   if (readflag == 0) {
      byteptr = buf;
      memset(buf, 0, bufsize);
   }
   else {
      bufsize = read(id, buf, bufsize);
      byteptr = buf;
   }
   bitoffset = 0;
   bytecount = 0;
   resetflag = 0;
}
