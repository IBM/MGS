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

#ifndef LZWBUF_H
#define LZWBUF_H
#include "Copyright.h"

#include "ImgUtil.h"
#include "Bitbuf.h"

class LZW
{
   char *buf;                    // output buffer pointer allocated outside and passed into program
   char *bufptr;                 //location of next data to be written
   char *decodestack;            // array to keep track of decode characters

   int decodeflag;
   int Bits, Clrcode, EOD, firstcode, Maxcode, Tblsize;
   int character;
   int stringcode;
   int savecode;
   int currentcodebits;
   int Unused;
   int index;
   int nextcode;
   int newcode;
   int nextbumpcode;
   int maxsize;
   int resetflag;

   struct dictionary
   {
      int codevalue;
      int parentcode;
      char character;
   } *dict;

   Bitbuf *output, *input;

   void InitDict(void);
   int findchildnode( int parentcode, int childcharacter );
   int decodestring(int count, int code);
   void reset(void);

   public:
      //constructor
      LZW(int idx, int bufsize, int xlzwcmp,
         int xBits, int xClr, int xEOD, const char* endstrx = "");
      //destructor
      ~LZW();

      void compress(unsigned char* inbuf, int length);
      int decompress(unsigned char* outbuf);
      void setreset(void){ resetflag = 1; }
};
#endif
