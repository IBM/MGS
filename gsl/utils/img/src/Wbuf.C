// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
/*
  This program is to buffer write outputs.
  Encoding codes:
  0 - none
  1 - 12 bits per component, ASCII85
  2 -  8 bits per component, ASCIIHEX
  3 - 12 bits per component, ASCIIHEX
  4 -  8 bits per component, ASCII85
*/
// #include <io.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "ImgUtil.h"
#include "Wbuf.h"

wbuf::wbuf(int idx, int sizex, int comprx, const char* endstrx)
{
   id = idx;
   size = outsize = sizex;
   compr = comprx;
   strcpy(endstr,endstrx);
   switch (compr) {
      case 4:                    //8 bits per component ASCII85 encoding
      case 1:                    //12 bit per component ASCII85 encoding
         size = (size>>4)<<4;    //make size divisible by 16
         buf = new unsigned char[size<<2];
         buf2 = buf+(size<<1);
         strcpy(endstr,"~>");
         break;
      case 3:                    // 12 bit per component ASCIIHEX encoding
         size = (size>>2)<<2;    // make size divisible by 4
      case 2:                    // ASCIIHEX encoding
      {
         // set up ascii table to convert binary to ascii
         char hexcode[16]= {
            '0','1','2','3','4','5','6','7',
            '8','9','A','B','C','D','E','F'
         };
         ascii = new unsigned short[256];
         for (i=0;i<256;i++)
            ascii[i] = (hexcode[0x0f&i]<<8) | hexcode[i>>4];
         buf = new unsigned char[size<<2];
         buf2 = buf+(size<<1);
         strcpy(endstr,">");
      }
      break;
      default:
         buf = new unsigned char[size];
   }
   if (buf==NULL) error("Could not allocate space for output buffer.\n");
   i = 0;
   finish = 0;
}


void wbuf::wput(void* ptr,int len)
{
   int j,k;
   while(len>0 || ptr == NULL) {
      if(i+len < size) {
         memcpy(&buf[i],ptr,len);
         i += len;
         break;
      }
      else {
         dfltflag = 0;
         if (ptr != NULL) memcpy(&buf[i],ptr,size-i);
         switch (compr) {
            case 1:              //compress 12 bit data using ASCII85Encode filter
            {
               unsigned short *sptr;
               unsigned long lnum;
               // first compact 16 bit data to 12 bit data.
               for (j=0,k=0;j<size;j+=4,k+=3) {
                  sptr = (unsigned short*)&buf[j];
                  (*sptr) <<= 4;
                  buf2[k] = buf[j+1];
                  buf2[k+1] = buf[j] | buf[j+3];
                  buf2[k+2]  = buf[j+2];
               }
               // convert 4 binary bytes to 5 ascii bytes
               outsize = k;
               for(j=0,k=0;k<outsize;k+=4) {
                  //reverse 4 bytes
                  buf[j+3] = buf2[k];
                  buf[j] = buf2[k+3];
                  buf[j+2] = buf2[k+1];
                  buf[j+1] = buf2[k+2];
                  // convert to 5 ASCII chars
                  lnum = *((unsigned long*)(&buf[j]));
                  if (lnum!=0 || ((k+4)>=outsize && finish>0) ) {
                     buf[j] = 33 + lnum/(85ul*85*85*85);
                     lnum %= 85ul*85*85*85;
                     buf[j+1] = 33 + lnum/(85*85*85);
                     lnum %= 85*85*85;
                     buf[j+2] = 33 + lnum/(85*85);
                     lnum %= 85*85;
                     buf[j+3] = 33 + lnum/85;
                     buf[j+4] = 33 + lnum%85;
                     j += 5;
                  }
                  else {
                     buf[j] = 'z';
                     j++;
                  }
               }
               outsize = j-finish;
            }
            break;
            case 2:              // compress with standard ascii
            {
               memcpy(buf2,buf,size);
               unsigned short* sptr;
               sptr = (unsigned short*)buf;
               for (j=0;j<size;j++)
                  sptr[j] = ascii[buf2[j]];
               outsize = j<<1;
            }
            break;
            case 3:              //compress 12 bit data using ASCIIHEX encode filter
            {
               unsigned short *sptr;
               // first compact 16 bit data to 12 bit data.
               for (j=0,k=0;j<size;j+=4,k+=3) {
                  sptr = (unsigned short*)&buf[j];
                  (*sptr) <<= 4;
                  buf2[k] = buf[j+1];
                  buf2[k+1] = buf[j] | buf[j+3];
                  buf2[k+2]  = buf[j+2];
               }
               // compress with standard ascii
               sptr = (unsigned short*)buf;
               for (j=0;j<k;j++)
                  sptr[j] = ascii[buf2[j]];
               outsize = j<<1;
            }
            break;
            case 4:              //compress 8 bit data using ASCII85 encode filter
               union
               {
                  unsigned long lnum;
                  struct
                  {
                     unsigned char byte0, byte1, byte2, byte3;
                  }b;
               }u;

               memcpy(buf2,buf,size);
               for(j=0,k=0;k<size;k+=4) {
                  //reverse 4 bytes
                  u.b.byte3 = buf2[k];
                  u.b.byte0 = buf2[k+3];
                  u.b.byte2 = buf2[k+1];
                  u.b.byte1 = buf2[k+2];
                  // convert to 5 ASCII chars
                  if (u.lnum!=0 || ((k+4)>=size && finish>0) ) {
                     buf[j] = 33 + u.lnum/(85ul*85*85*85);
                     u.lnum %= 85ul*85*85*85;
                     buf[j+1] = 33 + u.lnum/(85*85*85);
                     u.lnum %= 85*85*85;
                     buf[j+2] = 33 + u.lnum/(85*85);
                     u.lnum %= 85*85;
                     buf[j+3] = 33 + u.lnum/85;
                     buf[j+4] = 33 + u.lnum%85;
                     j += 5;
                  }
                  else {
                     buf[j] = 'z';
                     j++;
                  }
               }
               outsize = j-finish;
               break;
            default:
               outsize = size;
               dfltflag = 1;
         }
         if (dfltflag)
            size_t s=write(id,buf,outsize);
         else {
            //place eol's every 80 characters for encoding schemes
            char eol='\n';
            memcpy(buf2,buf,outsize);
            for (j=k=0; j<outsize; j+=80,k+=81) {
               memcpy(&buf[k],&buf2[j],80);
               buf[k+80] = eol;
            }
            size_t s=write(id,buf,outsize+outsize/80);
         }
         len -= size-i;
         ptr = (char*)ptr + size-i;
         i = 0;
      }
   }
}


wbuf::~wbuf()
{
   int fin[8]= {                 //number of ASCII bytes to dimish in last coding
      0,2,1,3,2,5,3,1
   };
   if (i > 0) {
      size=i;
      size_t s;
      switch (compr) {
         case 1:
            finish = size%16;
            if (finish > 0) memset(&buf[size],0,16-finish);
            finish = fin[finish>>1];
            wput(NULL,0);
            s=write(id,endstr,strlen(endstr));
            break;
         case 4:
            finish = size%4;
            if (finish > 0) {
               finish = 4-finish;
               memset(&buf[size],0,finish);
            }
            wput(NULL,0);
            s=write(id,endstr,strlen(endstr));
            break;
         case 2:
         case 3:
            wput(NULL,0);
            s=write(id,endstr,strlen(endstr));
            break;
         default:
            s=write(id,buf,i);
      }
   }
   if (compr==2 || compr==3) delete[] ascii;
   delete[] buf;
}
