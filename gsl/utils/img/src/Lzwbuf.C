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

/* This class implements LZW encoding and decoding into a buffer
  This is an LZW class which implements a more powerful version
  of the algorithm in 3 ways. It expands the maximum code size
  to 15 bits.  Second, it starts encoding with 9 bit codes, working
  its way up in bit size only as necessary.  Finally, it clears the
  dictionary when done.

  The dictionary data structure defines the dictionary.  Each entry in
  the dictionary has a code value.  This is the code emitted by the compressor.  Each
  The code is actually made up of two pieces:  a parent_code, and a
  character.  Code values of less than 256 are actually plain text codes.

Other global data structures:  The decodestack is used to reverse
std::strings that come out of the tree during decoding.  nextcode is the
next code to be added to the dictionary, both during compression and
decompression.  currentcodebits defines how many bits are currently
being used for output, and nextbumpcode defines the code that will
trigger the next jump in word size.

The compressor is short and simple.  It reads in new symbols one
at a time from the input file.  It then  checks to see if the
combination of the current symbol and the current code are already
defined in the dictionary.  If they are not, they are added to the
dictionary, and we start over with a new one symbol code.  If they
are, the code for the combination of the code and character becomes
our new code.  Note that in this enhanced version of LZW, the
encoder needs to check the codes for boundary conditions.

If bufsize < 0, LZW decompresses, otherwise compresses.
*/
#include"Lzwbuf.h"

//constructor
LZW::LZW(int idx, int bufsize, int comprx, int xBits, int xClr, int xEOD, const char* endstrx)
{
   decodeflag = 0;
   if (bufsize < 0) {
      bufsize = -bufsize;
      decodeflag = 1;
   }
   Bits = xBits;
   Clrcode = xClr;
   EOD = xEOD;
   firstcode = 258;
   Maxcode = (1<<Bits) - 1;
   Tblsize = 35023;
   Unused = -1;
   dict = new dictionary[Tblsize];
   if (dict==0) error("Could not allocate space for dictionary.\n");
   resetflag = 1;
   currentcodebits = 9;
   if (decodeflag) {
      maxsize = bufsize-Tblsize;
      if (maxsize <= 0) error("Increase output buffer size to > %d.\n",Tblsize);
      decodestack = new char[Tblsize];
      input = new Bitbuf(idx, -bufsize, comprx, endstrx);
   }
   else {
      output = new Bitbuf(idx, bufsize, comprx, endstrx);
   }
}


//destructor
LZW::~LZW(void)
{
   if (decodeflag) {
      delete[] decodestack;
      delete input;
   }
   else {
      output->putbits(savecode, currentcodebits);
      output->putbits(EOD, currentcodebits);
      delete output;
   }
   delete[] dict;
}


void LZW::InitDict(void)
{
   int i;
   nextcode = firstcode;
   for (i=0; i<Tblsize; i++)
      dict[i].codevalue = Unused;
   currentcodebits = 9;
   nextbumpcode = 511;
}


void LZW::compress(unsigned char* inbuf, int length)
{
   int i;
   int character;
   int stringcode;
   unsigned int index;
   i = 0;                        // i is count
   if (resetflag != 0) reset();
   if (savecode >= 0)
      stringcode = savecode;
   else {
      stringcode = inbuf[0];
      i++;
   }
   for (; i<length; i++) {
      character = inbuf[i];
      index = findchildnode(stringcode, character);
                                 // found
      if (dict[index].codevalue != Unused)
         stringcode = dict[index].codevalue;
      else {                     // not found
         dict[index].codevalue = nextcode++;
         dict[index].parentcode = stringcode;
         dict[index].character = (char)character;
         output->putbits(stringcode, currentcodebits);
         stringcode = character;
         if (nextcode > Maxcode) {
            output->putbits(Clrcode, currentcodebits);
            InitDict();
         }
         else if (nextcode > nextbumpcode) {
            // output->putbits(BUMP_CODE, currentcodebits ); not done in this version
            currentcodebits++;
            nextbumpcode <<= 1;
            nextbumpcode |= 1;
            // putc( 'B', stdout );
         }
      }
   }
   savecode = stringcode;
}


/*
   The file expander operates much like the encoder.  It has to
   read in codes, the convert the codes to a std::string of characters.
   The only catch in the whole operation occurs when the encoder
   encounters a CHAR+STRING+CHAR+STRING+CHAR sequence.  When this
   occurs, the encoder outputs a code that is not presently defined
   in the table.  This is handled as an exception.  All of the special
   input codes are handled in various ways.
   Returns actual length used. If return value = length, call should be
   repeated.
*/

int LZW::decompress(unsigned char* outbuf)
{
   int i;                        // i is count
   int character;
   int count;
   int nextclr;
   if (resetflag != 0) reset();
   nextclr = 0;
   i = 0;
   if (savecode < 0)
      savecode = input->getbits(currentcodebits);
   if (savecode == EOD)
      return(i);
   character = savecode;
   outbuf[i++] = savecode;
   while (i < maxsize) {
      if (currentcodebits > Bits)
         error("Code bit length of %d is too long.\n", currentcodebits);
      newcode = input->getbits(currentcodebits);
      if (newcode == EOD)
         return(-i);
      if (newcode == Clrcode) {
         InitDict();
         savecode = -1;
         nextclr = 0;
         return(i);
      }
      if (nextclr > 0) error("Next code should have been a Clrcode.\n");
      if (newcode >= nextcode) {
         decodestack[0] = (char)character;
         count = decodestring(1, savecode);
      }
      else
         count = decodestring(0, newcode);
      character = decodestack[count-1];
      while(count > 0)
         outbuf[i++] = decodestack[--count];
      dict[nextcode].parentcode = savecode;
      dict[nextcode].character = character;
      nextcode++;
      if (nextcode >= nextbumpcode) {
         if (nextcode >= Maxcode)
            nextclr = 1;
         else {
            currentcodebits++;
            nextbumpcode <<= 1;
            nextbumpcode |= 1;
         }
      }
      savecode = newcode;
   }
   return i;
}


void LZW::reset()
{
   // InitDict, read Clrcode, and reset Bitbuf
   savecode = -1;
   if (decodeflag) {
      input->setreset();
      InitDict();
      stringcode = input->getbits(currentcodebits);
      if (stringcode != Clrcode) error("Starting code in LZW file is not %d.\n",Clrcode);
   }
   else {
      output->setreset();
      output->putbits(Clrcode, currentcodebits);
      InitDict();
   }
   resetflag = 0;
}


int LZW::decodestring(int count, int code)
{
   while(code > 255) {
      decodestack[count++] = dict[code].character;
      code = dict[code].parentcode;
   }
   decodestack[count++] = (char)code;
   return(count);
}


/*
  This hashing routine is responsible for finding the table location
  for a std::string/character combination.  The table index is created
  by using an exclusive OR combination of the prefix and character.
  This code also has to check for collisions, and handles them by
  jumping around in the table.
  The return value is the index of the found code if it exists already
  or an unused index in the table.
*/

int LZW::findchildnode( int parentcode, int childcharacter )
{
   unsigned int index;
   unsigned int offset;

   index = (childcharacter<<(Bits-8)) ^ parentcode;
   if (index == 0)
      offset = 1;
   else
      offset = Tblsize-index;
   while (1) {
      if (dict[index].codevalue == Unused)
         return (index);
      if (dict[index].parentcode == parentcode &&
         dict[index].character == (char)childcharacter)
         return (index);
      if (index >= offset)
         index -= offset;
      else
         index += Tblsize-offset;
   }
}
