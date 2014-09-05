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

// Img.h: image class
#ifndef IMG_H
#include "Copyright.h"
#define IMG_H

#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

// Inserted, Ravi Rao, 8/28/02
// as this could not be found in any of the system .h
// files on AIX, and this is required by the code
// (ported from Windows).
#ifndef O_BINARY
#define O_BINARY 0
#endif

#include "Matrx.h"
#define BRITE .65

class Pal;

enum ImageType {NON, TIFF, WH, BMP, PCX ,RGB};
enum ROT {CC, R180, CW};

ImageType GetType(char* ifn);

class Image
{
   public:
      ImageType HdType;
      Pal *palpt;
      int id;                    // image handle
      unsigned short ru;         //resolution unit
      double xres, yres;         //x and y resolutions per unit
      short pi;                  // Photometric Interpretation

   protected:
      char *dt;                  // Date and time
      char *art;                 // Copywrite and artist
      char *cpyr;                // Copywrite
      char *doc;                 // Document name
      char *descr;               // Image description
      char *brcosh;              // brightness, contrast, sharpness
      unsigned short Sw;
      int w, h, sizepx;
      int expand;                // for automatic expansion of 4 bit grays
      unsigned short sampp;      //samples per pixel
      // Other parameters used in Line function
      unsigned long amt;         // Read and write (image) buffer size in bytes
      int cur;                   // current line of input image (0 for first line, -1 if none)
      int first,last;            // Number of first and last line in the image buffer
      unsigned char *lbuf;       // pointer to the image buffer for file operations
      int nbuf;                  // number of lines in buffer
      int nw;                    // number of bytes in a line.
      int nwe;                   // number of bytes in a line when 4bits/sample is expanded to 1 byte
      long *sbc;                 // strip byte count and offsets
      long *toff;
      int maxstrip;              // maximum strip size in bytes.
      unsigned short comp;
      unsigned short strippi;

   public:
      long imgoff;               // offset to start of image
      unsigned short *lut;       // transfer table
      long lutoff;               // transfer table offset
      short bitsps[4];           // Bits/sample
      short mem;                 // 0 indicates limited size, 1 indicates large image buffer
      Matrx ms;                  // Scanner matrix
      double wp[2];              // image white point
      double brite;
      // Methods
      long reads(int id,char* buf,long n);
      long readl(int id,char* buf,long n);
      long readt(int id,char* buf,long n);
      int GetSize(void){return sizepx;}
      void PutSize(int spx){sizepx = spx;}
      unsigned short GetSpp(void){return sampp;}
      void PutSpp(unsigned short samppx){sampp=samppx;}
      short* GetBits(void) {
         return bitsps;
      }                          //use to set bitsps too
      int width(void){return w;}
      void     width(int wi){w=wi;}
      int height(void){return h;}
      void     height(int hi){h=hi;}
      virtual char* Description(void){return descr;}
      virtual void Description(char*){}
      int fileid(void){return id;}
      virtual const char* GetAscii(short nx){return "";}
      virtual void ReadHeader(char*){}
      virtual void WriteHeader(char*){}
      virtual void PutWp(double *wpx){}
      virtual void GetWp(double *wpx){wpx[0]=wp[0];wpx[1]=wp[1];}
      virtual int Getnw(void){return (expand) ? nwe : nw;}
      virtual void PutChrm(double *chrmx){}
      virtual void GetChrm(double *chrmx){}
      virtual void PutRes(double resx){}
      virtual double GetRes(void){return 72.;}
      virtual unsigned short* GetTbl(void){return lut;}
      virtual void PutTbl(unsigned short *tblx){}
      virtual void RotHd(ROT rot) {
         unsigned short t; if(rot!=R180) {
            t=h; h=w; w=t;
         }
      }
      virtual void Expand(void); //sets up expansion of 4 bit grays
      virtual void IcopyHd(Image* pImage) {
         if (this->HdType == pImage->HdType) *this = *pImage;
         else {
            w = pImage->width();
            h = pImage->height();
            //      HdType = pImage->HdType;
            Sw = 0;
            sizepx = pImage->GetSize();
            sampp = pImage->GetSpp();
            palpt = pImage->palpt;
            for (int i=0;i<sampp;i++)
               bitsps[i] = pImage->bitsps[i];
         }
      }
      virtual void OpenLine(int num=0);
      virtual void* GetLine(int num=-1);
      virtual void* PutLine(int num=-1);
      virtual void UpLine(int num) {
         cur += num;
         if (cur>last) last = cur;
      }
      virtual void CloseLine();
      virtual short GetType(){return -1;}
      virtual void PutType(short xtype){}
      virtual Matrx GetMat(){return ms;}
      virtual void PutMat(Matrx mx){ms = mx;}
      virtual void PutBrite(double britex){brite = britex;}
      virtual double GetBrite(){return brite;}

      //Constructors
   public:
      Image() {
         palpt = NULL; lbuf = NULL; lut = NULL; toff = NULL; sbc = NULL;
         mem = 0; ru = 0; expand = 0; brite = BRITE;
      }
      Image(char* ifn);

      //Destructor
      //  virtual ~Image(){ delete[] lut; close(id);}
      virtual ~Image() {
         delete[] lut;
         if (toff != NULL) {
            delete[] toff;
            delete[] sbc;
         }
         close(id);
      }

      friend long BUFSIZE(Image* a);
};

Image *Header(ImageType it);

class WHImage:public Image
{
   public:

      //Methods
      virtual void ReadHeader(char* fname);
      virtual void WriteHeader(char* fname);

      //Constructors
      WHImage() {
         HdType = WH;
         pi = 1;
      }

      //Destructor
      virtual ~WHImage(){}
};

class RGBImage:public Image
{
   long img2, img3;
   unsigned char* tbuf;
   public:
      //Methods
      virtual void ReadHeader(char* fname);
      //virtual void WriteHeader(char* fname);
      virtual void OpenLine(int num=0);
      virtual void* GetLine(int num=-1);
      virtual void CloseLine();

      //Constructors
      RGBImage(){HdType = RGB; tbuf = NULL;}

      //Destructor
      virtual ~RGBImage(){}
};

class PCXImage:public Image
{
   unsigned short odd;
   unsigned char* tbuf;
   unsigned long bamt;
   public:

      //Methods
      virtual void ReadHeader(char* fname){}
      virtual void WriteHeader(char* fname);
      virtual void OpenLine(int num=0);
      virtual void* PutLine(int num=-1);
      virtual void CloseLine();

      //Constructors
      PCXImage() {
         odd = 0;
         bitsps[0] = 8;
         sizepx = sampp = 1;
         HdType = PCX;
      }

      //Destructor
      //  virtual ~PCXImage(){ delete[] lbuf; close(id); }  handled by Image destructor
};

class BMPImage:public Image
{
   long imgend;
   short bmptype;                // -1 no info, 0 OS/2, 1 Windows
   unsigned short odd;
   unsigned char* tbuf;

   public:
      //Methods
      virtual void ReadHeader(char* fname);
      virtual void WriteHeader(char* fname);
      virtual void OpenLine(int num=0);
      virtual void* PutLine(int num=-1);
      virtual void* GetLine(int num=-1);
      virtual void CloseLine();
      virtual void PutType(short xtype){bmptype = xtype;}
      virtual short GetType(){return bmptype;}

      //Constructors
      BMPImage() {
         char* env;
         env = getenv("BMPTYPE");
         if (env!=NULL) bmptype = atoi(env);
         else bmptype = 0;
         bitsps[0] = 8;
         sizepx = sampp = 1;
         HdType = BMP;
         pi = 1;
      }

      //Destructor
      //  virtual ~BMPImage(){ delete[] lbuf; close(id); }
};

class TiffImage:public Image
{
   short orient;                 // Orientation
   double chrm[8];               // display chromaticities and display white point
   long chr[3][2][2];            //raw display chromaticities
   long wpl[2][2];               // raw display white point
   short tiftype;                // -1 for no info, 0 for baseline tiff, 1 for our special tiff

   public:

      // Methods
      virtual const char* GetAscii(short nx);
      virtual void ReadHeader(char* fname);
      virtual void WriteHeader(char* fname);
      virtual void Description(char* descrx);
      virtual void RotHd(ROT rot);
      virtual void PutWp(double *wpx){memcpy(wp,wpx,sizeof(wp));}
      //  virtual void GetWp(double *wpx){memcpy(wpx,wp,sizeof(wp));}
      virtual void PutChrm(double *chrmx){memcpy(chrm,chrmx,sizeof(chrm));}
      virtual void GetChrm(double *chrmx){memcpy(chrmx,chrm,sizeof(chrm));}

      virtual void PutTbl(unsigned short *tblx);
      //  virtual void IcopyHd(Image* pImage) {*this = *(TiffImage*)pImage;}
      virtual void IcopyHd(Image* pImage);
      virtual short GetType(){return tiftype;}
      virtual void PutType(short xtype){tiftype=xtype;}
      virtual void PutRes(double resx){ xres = yres = resx;}
      virtual double GetRes(void){ return xres;}
      //  virtual Image* operator = (Image* pImage)
      //Constructors
      TiffImage() {
         char* env;
         env = getenv("TIFFDPI");
         if (env!=NULL) {
            ru = 2;
            xres = yres = atof(env);
         }
         else ru = 0;
         doc = descr = dt = art = cpyr = doc = dt = brcosh = NULL;
         sbc = toff = 0;
         tiftype = -1;
         imgoff = lutoff = w = h = 0;
         bitsps[0] = bitsps[1] = bitsps[2] = 8;
         sizepx = comp = strippi = sampp = 1;
         orient = pi = 1;
         HdType = TIFF;
         expand = 0;
      }

      //Destructor

      virtual ~TiffImage() {
         delete[] art;
         delete[] descr;
         delete[] doc;
         delete[] dt;
         delete[] brcosh;
         delete[] cpyr;
      }
};
#endif
