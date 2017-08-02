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

#include "Img.h"
#include "ImgUtil.h"
#include "Pal.h"
#include "Ini.h"
#include "Lzwbuf.h"

#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

inline long BUFSIZE(Image* a)
{
   return (a->mem > 0) ? a->mem : 400000/a->nw;
}


// read short
long Image::reads(int id,char* buf,long n)
{
   char i;
   size_t ct;
   ct = read(id,buf,n);
   if (Sw==0) return ct;

   printf("buf[0] = %d, buf[1] = %d\n", buf[0], buf[1]);

   while (n>0) {
      n -= 2;

      i = buf[n];
      buf[n] = buf[n+1];
      buf[n+1] = i;
      /*
      asm les bx,buf
      asm mov ax,es:[bx]
      asm xchg al,ah
      asm mov es:[bx],ax
      buf += 2;
      */
   }
   return ct;
}


// read long
long Image::readl(int id,char* buf,long n)
{
   char i;
   size_t ct;
   ct = read(id,buf,n);
   if (Sw==0) return ct;
   while (n>0) {
      n -= 4;
      i = buf[n];
      buf[n] = buf[n+3];
      buf[n+3] = i;
      i = buf[n+1];
      buf[n+1] = buf[n+2];
      buf[n+2] = i;
   }
   return ct;
}


// read tag
long Image::readt(int id,char* buf,long n)
{

   // printf("Sizeof long = %d; sizeof int = %d\n", sizeof(long), sizeof(int));
   struct tg {unsigned short TAG,TYP; long LEN,VAL;}
   *t;
   t = (tg*)buf;
   char i;
   size_t ct;
   ct = read(id,buf,n);
   if (Sw==0) {
      //      if ((t->LEN==1) && (t->TYP==3)) t->VAL >>= 16;
      //      if ((t->LEN==1) && (t->TYP==1)) t->VAL >>= 24;
      //      return ct;
   }
   else {
      while (n>4) {
         n -= 4;
         i = buf[n];
         buf[n] = buf[n+3];
         buf[n+3] = i;
         i = buf[n+1];
         buf[n+1] = buf[n+2];
         buf[n+2] = i;
      }
      while (n>0) {
         n -= 2;
         i = buf[n];
         buf[n] = buf[n+1];
         buf[n+1] = i;
      }
   }
   if ((t->LEN==1) && (t->TYP==3)) t->VAL >>= 16;
   if ((t->LEN==1) && (t->TYP==1)) t->VAL >>= 24;
   return ct;
}


ImageType GetType(char* ifn)
{
   int in=open(ifn,O_RDONLY|O_BINARY);
   if(in<0)
      error("Unable to open '%s'.\n",ifn);

   unsigned short w;
   size_t s=read(in,(char*)&w,2);
   close(in);

   if (w==0x4949 || w==0x4d4d)
      return TIFF;
   if (w==0x0a05) return PCX;
   if (w==0x4d42) return BMP;
   // if pdf file exists it is an RGB file
   char name[256];
   char* temptr;
   strcpy(name,ifn);
   temptr = strrchr(name,'.');
   if (temptr != NULL) {

      // THe following was old code, for MSVC++.
      // The _findfirst etc. don't work under GNU.
      // So this was replaced with the "stat" command, Ravi Rao, 8/22/02
      // struct _finddata_t ffblk;
      // int i;
      // temptr[0] = 0;
      // strcat(temptr,".pdf");
      // i = _findfirst(name,&ffblk);
      // if (i == 0)
      //   return RGB;

      // If ifn is "xyz.blah", create the name "xyz.pdf" and check
      // the pdf file exists
      temptr[0] = 0;
      strcat(temptr,".pdf");     // THis creates the filename with .pdf extension

      int fd, retcode;
      struct stat stat_buf;
      fd = open(name, 0);
      retcode = fstat(fd, &stat_buf);
      if (retcode != -1) {       // The pdf file exists
         return RGB;
      }
   }
   return WH;
}


Image *Header(ImageType it)
{
   Image *pImage=0;
   switch(it) {
      case NON:
         pImage = new WHImage();
         pImage->HdType = NON;
         break;

      case WH:
         pImage = new WHImage();
         break;

      case TIFF:
         pImage = new TiffImage();
         break;

      case PCX:
         pImage = new PCXImage();
         break;

      case BMP:
         pImage = new BMPImage();
         break;

      case RGB:
         pImage = new RGBImage();
         break;

      default:
         error("Unknown image type.\n");
         break;
   }
   return pImage;
}


void Image::OpenLine(int num)
{
   //header read/write is complete, file pointer is at beginning of data
   //cannot handle read and write, just one or the other.
   //set up buffer
   if (sizepx > 0) nw=w*sizepx;
   else {
      int n, ct;
      for(n=ct=0;n<sampp;n++) ct+=bitsps[n];
      nw = (ct*w+7)/8;           // padded at end of row
   }
   mem = num;
   nbuf = BUFSIZE(this);         //based on nw
   if (comp == 5) {
      int i,j, lbufi;
      amt = (long)h*nw;
      nwe = nw;
      lbuf = new unsigned char[amt];
      LZW *lzw;
      lzw = new LZW(id,-(maxstrip+36000),0,12,256,257);
      lbufi = 0;
      for (j=0; j<strippi; j++) {
         imgoff = lseek(id,toff[j],SEEK_SET);
         lzw->setreset();        //reset so that buffer is assumed empty
         i = lzw->decompress(&lbuf[lbufi]);
         while (i > 0) {
            lbufi += i;
            i = lzw->decompress(&lbuf[lbufi]);
         }
         lbufi -= i;
      }
      first = 0;
      last = h;
      cur = -1;
      delete lzw;
   }
   else if (comp == 32773) {     // PackBits compression
      int i,j,k,n;
      unsigned char* stripptr, *lbufptr;
      amt = (long)h*nw;
      nwe = nw;
      lbuf = new unsigned char[amt+maxstrip];
      if (lbuf == NULL) error("Could not allocate space for uncompressed image.\n");
      stripptr = lbuf+amt;
      lbufptr = lbuf;
      for (i=0; i<strippi; i++) {
         imgoff = lseek(id,toff[i],SEEK_SET);
         size_t s=read(id,stripptr,sbc[i]);
         // now decompress at stripptr to lbufptr
         n = 0;                  // initial strip byte count
         while (n < sbc[i]) {
            j = 0;
            while (j<nw) {
               k = stripptr[n];
               if (k < 128) {
                  memcpy(&lbufptr[j],&stripptr[n+1],k+1);
                  j += k+1;
                  n += k+2;
               }
               else if (k != 0x80) {
                  memset(&lbufptr[j],stripptr[n+1],257-k);
                  j += 257-k;
                  n += 2;
               }
               else n++;         //noop
            }
            lbufptr += j;
         }
         if (n != sbc[i]) printf("Warning, strip count not valid.\n");
      }
      first = 0;
      last = h;
      cur = -1;
   }
   else {
      amt = (long)nbuf*nw;
      if (expand) {
         lbuf = new unsigned char[amt<<1];
         nwe = nw<<1;
      }
      else {
         lbuf = new unsigned char[amt];
         nwe = nw;
      }
      if (lbuf==NULL) error("Could not allocate space for read/write buffer.");
      imgoff = lseek(id,0,SEEK_CUR);
      first = 0;
      last = -1;
      cur = -1;
   }
}


void* Image::GetLine(int num)
{
   size_t n;
   cur = (num < 0) ? cur+1 : num;
   if(cur>last || cur<first) {   // need to get more data
      if (cur > last && cur != h-1) {
         lseek(id,imgoff+(long)nw*cur,SEEK_SET);
         n = read(id,&lbuf[0],amt);
         if ((unsigned int) n<amt)
            nbuf = n/nw;
         if (nbuf<=0) error("Unexpected end of read file.");
         first = cur;
         last = cur+nbuf-1;
         int i;
         if (pi == 0)
            for (i=0;i<n;i++)
               lbuf[i] = ~lbuf[i];
      }
      else {                     //going backwards
         if (cur-(nbuf-1) < 0 )
            nbuf = cur+1;
         lseek(id,imgoff+(long)nw*(cur-(nbuf-1)),SEEK_SET);
         n = read(id,&lbuf[0],amt);
         first = cur-(nbuf-1);
         last = cur;
         int i;
         if (pi == 0)
            for (i=0;i<n;i++)
               lbuf[i] = ~lbuf[i];
      }
      if (expand) {              // enlarge 4 bit gray data
         int i, end, lend;
         unsigned char j;
         end = n;
         lend = (end<<1)-1;
         for (i=end-1;i>0;i--,lend-=2) {
            //          lbuf[lend-1] = (lbuf[i]|0x0F);
            //          lbuf[lend] = ((lbuf[i]|0xF0)<<4);
            lbuf[lend-1] = lbuf[i]&0xF0;
            lbuf[lend] = (lbuf[i]&0x0F)<<4;
         }
         // first pixel
         j = lbuf[0];
         //      lbuf[0] = (j|0x0F);
         //      lbuf[1] = ((j|0xF0)<<4);
         lbuf[0] = j&0xF0;
         lbuf[1] = (j&0x0F)<<4;
      }
   }
   return &lbuf[(long)(cur-first)*nwe];
}


void* Image::PutLine(int num)
{
   cur = (num < 0) ? cur+1 : num ;
   if(cur >= first+nbuf) {       // need to write data
      if (expand) {              // compress 4 bit gray data
         int i, end, l;
         end = (cur-first)*nw;
         // first two bytes
         i = lbuf[0];
         l = lbuf[1];
         if (pi == 0) {
            lbuf[0] = ~((i&0xF0)+((l&0xF0)>>4));
            for (i=1,l=2;i<end;i++,l+=2)
               lbuf[i] = ~((lbuf[l]&0xF0) | ((lbuf[l+1]&0xF0)>>4));
         }
         else {
            lbuf[0] =(i&0xF0)+((l&0xF0)>>4);
            for (i=1,l=2;i<end;i++,l+=2)
               lbuf[i] = (lbuf[l]&0xF0) | ((lbuf[l+1]&0xF0)>>4);
         }
      }
      size_t s=write(id,&lbuf[0],(cur-first)*nw);
      first = cur;
   }
   return &lbuf[(long)(cur-first)*nwe];
}


void Image::CloseLine()
{
   if (expand) {                 // compress 4 bit gray data
      int i, end, l;
      end = (cur+1-first)*nw;
      // first two bytes
      i = lbuf[0];
      l = lbuf[1];
      if (pi == 0) {
         lbuf[0] = ~((i&0xF0)+((l&0xF0)>>4));
         for (i=1,l=2;i<end;i++,l+=2)
            lbuf[i] = ~((lbuf[l]&0xF0) | ((lbuf[l+1]&0xF0)>>4));
      }
      else {
         lbuf[0] =(i&0xF0)+((l&0xF0)>>4);
         for (i=1,l=2;i<end;i++,l+=2)
            lbuf[i] = (lbuf[l]&0xF0) | ((lbuf[l+1]&0xF0)>>4);
      }
   }
                                 //write excess if writeable
   size_t s=write(id,&lbuf[0],(long)(cur+1-first)*nw);
   delete[] lbuf;
}


void Image::Expand(void)         //sets up expansion of 4 bit grays
{
   if (sizepx==0 && sampp==1 && bitsps[0]==4) expand = 1;
}


void WHImage::ReadHeader(char* ifn)
{
   unsigned short usw,ush;
   id=open(ifn,O_RDONLY|O_BINARY);
   if(id<0) error("Unable to open '%s'.\n",ifn);

   if (HdType != NON) {
      size_t s=read(id,(char*)&usw,2);
      w = usw;
      if (w == 0) {              // file is Unix type
         //reverse order of 4 byte width and height
         union
         {
            struct { char c[4];}
            s;
            int w;
         }u;
         char a[4];
         lseek(id,0,SEEK_SET);
         s=read(id,a,4);
         u.s.c[0] = a[3]; u.s.c[1] = a[2]; u.s.c[2] = a[1]; u.s.c[3] = a[0];
         w = u.w;
         s=read(id,a,4);
         u.s.c[0] = a[3]; u.s.c[1] = a[2]; u.s.c[2] = a[1]; u.s.c[3] = a[0];
         h = u.w;
         imgoff = 8;
      }
      else {
         s=read(id,(char*)&ush,2);
         h = ush;
         imgoff = 4;
      }

      //    sizepx = (filelength(id)-imgoff)/((long)w*h);
      // This is Windows code, and uses the filelength function of io.h in MSVC++
      // Replace "filelength(id)" with "stat_buf.st_size"
      // as follows, Ravi Rao, 8/28/02

      struct stat stat_buf;
      fstat(id, &stat_buf);
      sizepx = (stat_buf.st_size-imgoff)/((long)w*h);

   }

   switch(sizepx) {
      case 4:
         bitsps[0] = 8;
         bitsps[1] = 8;
         bitsps[2] = 8;
         bitsps[3] = 8;
         sampp = 4;
         break;
      case 3:
         bitsps[0] = 8;
         bitsps[1] = 8;
         bitsps[2] = 8;
         sampp = 3;
         break;
      case 1:
         bitsps[0] = 8;
         sampp = 1;
         break;
      case 2:
         bitsps[0] = 5;
         bitsps[1] = 6;
         bitsps[2] = 5;
         sampp = 3;
         break;
      case 0:
         // if (8*(filelength(id)-imgoff)/((long)w*h) == 1)
         // This is Windows code, and uses the filelength function of io.h in MSVC++
         // Replace "filelength(id)" with "stat_buf.st_size"
         // as follows, Ravi Rao, 8/28/02

         struct stat stat_buf;
         fstat(id, &stat_buf);
         if (8*(stat_buf.st_size-imgoff)/((long)w*h) == 1) {
            sampp = 1;
            bitsps[0] = 1;
            break;
         }
         else if (2*(stat_buf.st_size-imgoff)/((long)w*h) == 1) {
            sampp = 1;
            bitsps[0] = 4;
            break;
         }
      default:
         error("Don't recognize type of pixels in '%s'.\n",ifn);
   }
}


void WHImage::WriteHeader(char* ofn)
{
   id=open(ofn,O_CREAT|O_WRONLY|O_BINARY|O_TRUNC,S_IWRITE);
   if(id<0) error("Unable to open %s for writing.\n",ofn);
   size_t s=write(id,(char*)&w,2);
   s=write(id,(char*)&h,2);
}


void RGBImage::ReadHeader(char* ifn)
{
   char name[256];
   char* temptr;
   strcpy(name,ifn);
   temptr = strrchr(name,'.');
   if (temptr != NULL) {
      int i,n;
      char buf[5000], fname[256];
      strcpy(temptr,".pdf");
      //get width and height from .pdf file
      i = open(name,O_RDONLY|O_BINARY);
      if (i < 0) error("Unable to open '%s'.\n",name);

      // n = filelength(i);
      // This is Windows code, and uses the filelength function of io.h in MSVC++
      // Replace "filelength(id)" with "stat_buf.st_size"
      // as follows, Ravi Rao, 8/28/02
      struct stat stat_buf;
      fstat(i, &stat_buf);
      n = stat_buf.st_size;

      size_t s=read(i,buf,n);
      close(i);
      buf[n]=0;
      temptr = strstr(buf,"Image width---");
      if (temptr==NULL) error("Could not find Image width---.\n");
      sscanf(temptr+14,"%d",&w);
      temptr = strstr(temptr,"Image height---");
      sscanf(temptr+15,"%d",&h);
      temptr = strstr(temptr,"Image file---");
      sscanf(temptr+13,"%s",fname);

      // if (strstr(strupr(ifn),strupr(fname)) == NULL)
      //      printf("File name, %s, in .pdf doesn't agree with .rgb name.\n",fname);
      // The Above "strupr" is not portrable from MSVC++.
      // Changed, Ravi Rao to the following.
      // This didn't work, 8/28/02 -- come back to this later
      #ifdef TRY_AGAIN
      String Xstr, Ystr, UXstr, UYstr;
      Xstr = ifn;
      Ystr = fname;
      UXstr = upcase(Xstr);
      UYstr = upcase(Ystr);
      if (strstr(UXstr,UYstr) == NULL)
         printf("File name, %s, in .pdf doesn't agree with .rgb name.\n",fname);
      #endif

   }
   id=open(ifn,O_RDONLY|O_BINARY);
   if(id<0) error("Unable to open '%s'.\n",ifn);

   if (HdType != NON) {
      imgoff = 0;

      // sizepx = filelength(id)/((long)w*h);
      // This is Windows code, and uses the filelength function of io.h in MSVC++
      // Replace "filelength(id)" with "stat_buf.st_size"
      // as follows, Ravi Rao, 8/28/02

      struct stat stat_buf;
      fstat(id, &stat_buf);
      sizepx = stat_buf.st_size/((long)w*h);
   }
   switch(sizepx) {
      case 4:
         bitsps[0] = 8;
         bitsps[1] = 8;
         bitsps[2] = 8;
         bitsps[3] = 8;
         sampp = 4;
         break;
      case 3:
         bitsps[0] = 8;
         bitsps[1] = 8;
         bitsps[2] = 8;
         sampp = 3;
         break;
      case 1:
         bitsps[0] = 8;
         sampp = 1;
         break;
      default:
         error("Don't recognize type of pixels in '%s'.\n",ifn);
   }
}


/*
void RGBImage::WriteHeader(char* ifn)
{
  error("Writing an RGBImage header not implemented yet.\n");
}
*/
void RGBImage::OpenLine(int num)
{
   //header read/write is complete, file pointer is at beginning of data
   //cannot handle write, just read
   //set up buffer(s)
   imgoff = 0;
   img2 = (long)w*h;
   img3 = img2<<1;
   nw=w*sizepx;
   mem = num;
   nbuf = BUFSIZE(this);
   amt = (long)nbuf*w;
   if (sizepx > 1) tbuf = new unsigned char[amt];
   lbuf = new unsigned char[(long)nbuf*nw];
   if (lbuf==NULL) error("Could not allocate space for read/write buffer.");
   first = 0;
   last = -1;
   imgoff = 0;
   cur = -1;
}


void* RGBImage::GetLine(int num)
{
   size_t n;
   int i,j;
   cur = (num < 0) ? cur+1 : num;
   if(cur>last || cur<first) {   // need to get more data
      if (sizepx == 1) {
         lseek(id,imgoff+(long)w*cur,SEEK_SET);
         n = read(id,lbuf,amt);
      }
      else {
         lseek(id,img3+(long)w*cur,SEEK_SET);
         n = read(id,tbuf,amt);
         // interleave rgb
         for (i=0,j=0; i<n; i++,j+=3)
            lbuf[j+2] = tbuf[i];
         lseek(id,img2+(long)w*cur,SEEK_SET);
         n = read(id,tbuf,amt);
         for (i=0,j=0; i<n; i++,j+=3)
            lbuf[j+1] = tbuf[i];
         lseek(id,imgoff+(long)w*cur,SEEK_SET);
         n = read(id,tbuf,amt);
         for (i=0,j=0; i<n; i++,j+=3)
            lbuf[j] = tbuf[i];
      }
      if ((unsigned long)n<amt)
         nbuf = n/w;
      if (nbuf<=0) error("Unexpected end of read file.");
      first = cur;
      last = cur+nbuf-1;
   }
   return &lbuf[(long)(cur-first)*nw];
}


void RGBImage::CloseLine()
{
   delete[] tbuf;
   delete[] lbuf;
}


void TiffImage::PutTbl(unsigned short *tblx)
{
   lut = new unsigned short[sampp*256];
   if (lut==NULL) error("TiffImage: Cannot allocate space for lut.");
   if (pi==0 && sampp==1) {      //invert table for gray values
      unsigned short i;
      for (i=0;i<256;i++)
         tblx[i]= 65535-tblx[i];
   }
   memcpy(&lut[0],tblx,sampp*2*256);
}


void TiffImage::Description(char* descrx)
{
   if (descr != NULL)
      delete[] descr;
   descr = new char[(unsigned short)strlen(descrx)+1];
   strcpy(descr,descrx);
}


const char* TiffImage::GetAscii(short nx)
{
   switch(nx) {
      case 1:
         return dt;
      case 2:
         return art;
      case 3:
         return descr;
      case 4:
         return brcosh;
      default:
         return NULL;
   }
}


void TiffImage::RotHd(ROT rot)
{
   // rot= CC, R180, or CW
   unsigned short temp;
   int rotcc[9]={0,6,5,8,7,4,3,2,1};
   int rotcw[9]={0,8,7,6,5,2,1,4,3};
   int rot180[9]={0,3,4,1,2,7,8,5,6};
   if (rot != R180) {
      temp = h;
      h = w;
      w = temp;
   }
   if (orient<1 || orient>8)
      error("Invalid orientation in Tiff file.\n");
   switch (rot) {
      case CC:
         orient = rotcc[orient];
         break;
      case CW:
         orient = rotcw[orient];
         break;
      case R180:
         orient = rot180[orient];
         break;
      default:
         error("Error in rotation spec.\n");
   }
}


#define tag (entry.TAG)
#define typ (entry.TYP)
#define len (entry.LEN)
#define value (entry.VAL)
#define seek(x) lseek(id,x,SEEK_SET)
#define here()  lseek(id,0,SEEK_CUR)
#define RATIONAL 5
#define LONG 4
#define SHORT 3
#define ASCII 2
#define ifd(a,b,c,d) tag=a; typ=b; len=(c); value=(d); s=write(id,&tag,12)

void TiffImage::ReadHeader(char* ifn)
{
   struct {unsigned short TAG,TYP; long LEN,VAL;}
   entry;
   long off,sbcount; 
   long stripoff=0;
   short i,j;
   unsigned short count;
   char buf[1536];

   id=open(ifn,O_RDONLY|O_BINARY);
   if(id<0) error("Unable to open '%s'.\n",ifn);
   unsigned short x;
   unsigned short n,ct;
   size_t s=read(id,&x,2);
   if (x==0x4949)                //Intel order
      Sw = 1;
   else if (x==0x4d4d)           // Motorola order
   #ifdef LINUX
      Sw = 1;                    // For unix machine
   #else
   Sw = 0;
   #endif
   else error("Image: input file not TIFF.");

   reads(id,(char*)&x,2);

   // std::cout << "Sw = " << Sw << std::cout << "; x = " << x << std::endl;

   //   if(((char*) &x)[1]!=42) {
   if(x!=42) {
      error("Image: Second integer in TIFF file no good.");
   }
   readl(id,(char*)&off,4);
   seek(off);
   reads(id,(char*)&count,2);
   unsigned short pc=1;
   short otag=0;
   unsigned int res[2];
   //   for (n=0;n<((char*)&count)[1];n++)
   for (n=0;n<count;n++) {
      readt(id,(char*)&entry,sizeof(entry));
      if (tag<otag) printf("TIFF tags are not in increasing order.\n");

      // Ravi Rao
      // std::cout << "Tag = " << tag << std::endl;
      switch(tag) {
         case 0x100:             // image width
            w=(int)value;
            break;
         case 0x101:             // image height
            h=(int)value;
            break;
         case 0x102:             // bits per sample
            if(len==1) bitsps[0]=(unsigned short)(value);
            else {
               off=here();
               seek(value);
               reads(id,(char*)bitsps,8);
               seek(off);
            }
            break;
         case 0x103:             // compression
            comp=(unsigned short)value;
            break;
         case 0x106:             // photometric interpretation
            pi=(unsigned short)value;
            break;
         case 0x107:             // thresholding
            bitsps[0] = 1;
            sampp = 1;
            break;
         case 0x10d:             // Document name
            off=here();
            seek(value);
            doc=new char[(unsigned short)len+1];
            if (doc==NULL) error("Not enough space for doc in tiffread.");
            s=read(id,&doc[0],(unsigned short)len);
            doc[(unsigned short)len] = 0;
            seek(off);
            break;
         case 0x10e:             // Image description
            off=here();
            seek(value);
            descr=new char[(unsigned short)len+1];
            if (descr==NULL) error("Not enough space for descr in tiffread.");
            s=read(id,&descr[0],(unsigned short)len);
            descr[(unsigned short)len] = 0;
            //      sscanf(&descr[0],"%5s%lg%*1s%lg%",wpbuf,&wpdescr[0],&wpdescr[1]);
            seek(off);
            break;
         case 0x111:             // img offset
            imgoff=value;
                                 //save strip offset location and first offset
            if ((unsigned short)len>1) {
               off=here();
               stripoff = value;
               strippi = (unsigned short)len;
               seek(value);
               if (typ==4)
                  readl(id,(char*)&imgoff,4);
               else error("Tiffread: strip offsets are not long.");
               seek(off);
            }
            break;
         case 0x112:             // orientation
            orient=(short)value;
            break;
         case 0x115:             // Samples/pixel
            sampp=(unsigned short)value;
            break;
         case 0x117:             // Strip Byte Count
            sbcount=value;
            if (strippi>1) {     //check to see if strips are consecutive
               toff = new long[strippi];
               if (toff==NULL) error("Tiffread: cannot allocate space for strip offsets.");
               if (typ==4) {
                  sbc = new long[strippi];
                  if (sbc==NULL) error("Tiffread: cannot allocate space for strip counts.");
                  off=here();
                  seek(stripoff);
                  readl(id,(char*)&toff[0],sizeof(long)*strippi);
                  seek(sbcount);
                  readl(id,(char*)&sbc[0],sizeof(long)*strippi);
                  for (i=1,maxstrip = sbc[0]; i<strippi; i++) {
                     if(toff[i]-toff[i-1] != sbc[i-1] && comp!=5)
                        error("Tiffread: cannot handle nonconsecutive strips.");
                     if (sbc[i] > maxstrip) maxstrip = sbc[i];
                  }
               }
               else {
                  short *ssbc;
                  int halfs;
                  sbc = new long[strippi];
                  halfs = strippi>>1;
                  ssbc = ((short*)sbc) + strippi;
                  if (sbc==NULL) error("Tiffread: cannot allocate space for strip counts.");
                  off=here();
                  seek(stripoff);
                  readl(id,(char*)&toff[0],sizeof(long)*strippi);
                  seek(sbcount);
                  reads(id,(char*)&ssbc[0],sizeof(short)*strippi);
                  sbc[0] = maxstrip = (long)ssbc[0];
                  for (i=1;i<strippi;i++) {
                     sbc[i] = (long)ssbc[i];
                     if((short)(toff[i]-toff[i-1]) != sbc[i-1] && comp!=5)
                        error("Tiffread: cannot handle nonconsecutive strips.");
                     if (sbc[i] > maxstrip) maxstrip = sbc[i];
                  }
               }
               seek(off);
            }
            break;
         case 0x11a:             //x resolution
            off=here();
            seek(value);
            readl(id,(char*)&res[0],8);
            if (res[1] != 0)
               xres = (double)res[0]/res[1];
            else
               xres = 0;
            ru = 2;              //set inches as default
            seek(off);
            break;
         case 0x11b:             //y resolution
            off=here();
            seek(value);
            readl(id,(char*)&res[0],8);
            if (res[1] != 0)
               yres = (double)res[0]/res[1];
            else
               yres = 0;
            ru = 2;              //set inches as default
            seek(off);
            break;
         case 0x11c:             // Planar configuration
            pc=(unsigned short)value;
            break;
         case 0x123:             // Gray Response Curve
            off=here();
            seek(value);
            lutoff = value;
            lut=new unsigned short[(unsigned short)len];
            if(lut==NULL) error("tiffread: No space for lut.");
            reads(id,(char*)&lut[0],(unsigned short)len*2);
            seek(off);
            break;
         case 0x128:             // resolution unit
            ru = (unsigned short)value;
            break;
         case 0x12d:             // TransferFunction
            off=here();
            seek(value);
            lutoff = value;
            lut=new unsigned short[(unsigned short)len];
            if(lut==NULL) error("tiffread: No space for lut.");
            reads(id,(char*)&lut[0],(unsigned short)len*2);
            seek(off);
            break;
         case 0x132:             // Date time
            off=here();
            seek(value);
            dt=new char[(unsigned)len+1];
            if(dt==NULL) error("no space for dt in tiffread");
            s=read(id,dt,(unsigned short)len);
            seek(off);
            break;
         case 0x13b:             // Copyright and artist
            off=here();
            seek(value);
            art=new char[(unsigned)len+1];
            if(art==NULL) error("no space for art in tiffread");
            s=read(id,art,(unsigned short)len);
            seek(off);
            break;
         case 0x13e:             // have standard tiff
            tiftype=0;
            off=here();
            seek(value);
            readl(id,(char*)&wpl[0][0],sizeof(wpl));
            chrm[6]=(double)wpl[0][0]/wpl[0][1];
            chrm[7]=(double)wpl[1][0]/wpl[1][1];
            seek(off);
            break;
         case 0x13f:             // have standard tiff
            off=here();
            seek(value);
            readl(id,(char*)&chr[0][0][0],sizeof(chr));
            chrm[0]=(double)chr[0][0][0]/chr[0][0][1];
            chrm[1]=(double)chr[0][1][0]/chr[0][1][1];
            chrm[2]=(double)chr[1][0][0]/chr[1][0][1];
            chrm[3]=(double)chr[1][1][0]/chr[1][1][1];
            chrm[4]=(double)chr[2][0][0]/chr[2][0][1];
            chrm[5]=(double)chr[2][1][0]/chr[2][1][1];
            seek(off);
            break;
         case 0x140:             // Color Map for palettized images
            off=here();
            seek(value);
            if(len!=768) error("ColorMap unexpected size.");
            reads(id,buf,1536);
            palpt = new Pal;
            if (palpt==NULL) error("Could not allocate space for Pal.");
            for (i=1,j=0;i<512;i+=2) {
               palpt->pall[j++] = buf[i];
               palpt->pall[j++] = buf[512+i];
               palpt->pall[j++] = buf[1024+i];
            }
            seek(off);
            break;
         case 0x80cf:            // Scanner matrix and white point
            tiftype=1;
            off=here();
            seek(value);
            s=read(id,buf,(unsigned short)len);
            sscanf(buf,"%lg%lg%lg%lg%lg%lg%lg%lg%lg%lg%lg",
               &ms.e[0][0],&ms.e[0][1],&ms.e[0][2],
               &ms.e[1][0],&ms.e[1][1],&ms.e[1][2],
               &ms.e[2][0],&ms.e[2][1],&ms.e[2][2],
               &wp[0],&wp[1]);
            seek(off);
            break;
         case 0x80d1:            // brcosh parameters
            char* tptr;
            char temstr[49];
            off=here();
            seek(value);
            brcosh=new char[48];
            if(brcosh==NULL) error("no space for brcosh in tiffread");
            s=read(id,brcosh,48);
            tptr = strstr(brcosh,"B=");
            if (tptr != NULL) {
               strcpy(temstr,tptr);
               brite = atof(strtok(temstr+2," ,"));
            }
            seek(off);
            break;
         case 0x8298:            // Copyright
            off=here();
            seek(value);
            cpyr=new char[(unsigned)len+1];
            if(cpyr==NULL) error("no space for copyright\n");
            s=read(id,cpyr,(unsigned short)len);
            seek(off);
            break;
         default:
            //      printf("Tiff tag %#06x (type=%2d, count=%ld, value=%ld) not processed.\n",
            //                   tag,typ,len,value);
            break;
      }
      otag = tag;
   }

   s=read(id,&amt,4);              // long but must be zero for now
   if (amt != 0) error("tiffread: More than one strip offset, or wrong tag count.");
   if (w==0) error("tiffread: width field missing.");
   if (h==0) error("tiffread: height field missing.");
   if (bitsps[0]==0) error("tiffread: bits/sample field missing.");
   if (comp!=1 && comp!=5 && comp!=32773) {
      printf("Width: %d    Height: %d\n",w,h);
      error("tiffread: cannot handle %d compression.\n",comp);
   }
   if (pi==4) error("tiffread: cannot handle photometric interpretation of 4.");
   if (pi==3 && (sampp!=1 || palpt==NULL))
      error("tiffread: ColorMap not present or sample size incorrect for PI=3.");
   if (pc!=1 && sampp>1) error("tiffread: file must be rgb interleaved.");
   if (imgoff==0) error("tiffread: Image offset field missing.");
   if (sampp==0) error("tiffread: Samples/pixel field missing.");
                                 //  convert csc to ms
   if((tiftype==0) && (sampp!=1)) {
      MatrxChrom(ms,chrm);
      wp[0]= .3457;
      wp[1]= .3587;
   }
   lseek(id,imgoff,SEEK_SET);    // position file pointer to start of image
   for(n=ct=0;n<sampp;n++) ct+=bitsps[n];
   sizepx = ct/8;
                                 //in this case bitsps must be used directly
   if (ct != sizepx*8) sizepx = 0;
}


void TiffImage::IcopyHd(Image* pImage)
{
   if (pImage->HdType==TIFF) {
      *this = *(TiffImage*)pImage;
      comp = 1;
   }
   else {
      w = pImage->width();
      h = pImage->height();
      sizepx = pImage->GetSize();
      bitsps[0] = pImage->bitsps[0];
      bitsps[1] = pImage->bitsps[1];
      bitsps[2] = pImage->bitsps[2];
      sampp = pImage->GetSpp();
   }
}


void TiffImage::WriteHeader(char* ofn)
{
   unsigned short x,k,bps,pal=0;
   long off;
   struct {unsigned short TAG,TYP; long LEN,VAL;}
   entry;
   id=open(ofn,O_CREAT|O_WRONLY|O_BINARY|O_TRUNC,S_IWRITE);
   if(id<0) error("Unable to open %s for writing.\n",ofn);
   tag= 0x4949;                  // intel order
   typ=42;                       // version
   len=8;                        // header off
   size_t s=write(id,&tag,8);

   // determine number of tags, nifd
   unsigned short nifd=13;       // number of ifd's;
                                 //image is palettized
   if (palpt != NULL && sampp ==1)
      pal = 1;
   if (doc!=NULL) nifd++;
   if (pal==0 && lut!=NULL)  nifd++;
   if (sampp!=1) {
      if(tiftype==1) nifd++;
   }
   else if (tiftype==0) nifd += 2;
   if (pal!=0) nifd++;
   if (brite!=BRITE || brcosh!=NULL) nifd++;
   if (descr == NULL) nifd--;
   if (art != NULL) nifd++;
   if (cpyr != NULL) nifd++;
   if (ru != 0) nifd += 3;

   s=write(id,&nifd,2);
   off=8+2+nifd*12+4;
   ifd(0xfe,LONG,1,0);           // New subfile type
   ifd(0x100,LONG,1,w);          // Image width
   ifd(0x101,LONG,1,h);          // Image height
   long bitsps_off=0;
   if (sampp>1) {
      ifd(0x102,SHORT,sampp,off);// Bits per sample
      bitsps_off=off;
      off+=2*sampp;              //pointer to bits per sample for each sample
   }
   else {                        // Bits for one sample
      ifd(0x102,SHORT,1,bitsps[0]);
   }
   ifd(0x103,SHORT,1,1);         // No compression
   //(0=white low, 1=black low, 2=rgb, 3=palette color)Photometric Interpretation
   if (pal==1) {                 //image is palettized
      ifd(0x106,SHORT,1,3);
   }
   else {                        //brackets needed for macro
      // ifd(0x106,SHORT,1,(sampp==1) ? pi:2);
      if(sampp==1 && pi>1) pi = 1;
      ifd(0x106,SHORT,1,pi);
   }
   //temporarily added
   //if (sampp==1) {ifd(0x107,SHORT,1,1);}   // Thresholding
   long doc_off=0;
   if (doc!=NULL) {
                                 // Document name.
      ifd(0x10d,ASCII,x=strlen(&doc[0])+1,off);
      doc_off=off; off+=x;
      //  nifd++;
   }
   long descr_off=off;
   if (descr != NULL) {
                                 // Image description.
      ifd(0x10e,ASCII,x=strlen(descr)+1,off);
      off+=x;
   }
   ifd(0x111,LONG,1,0);          // Strip offset (offset to image)
   long stripoff_loc=here()-4;
   ifd(0x112,SHORT,1,orient);    // Orientation
   ifd(0x115,SHORT,1,sampp);     // Samples per pixel
   ifd(0x116,LONG,1,h);          // Rows per strip (i.e. h for one strip image)
   bps=7;                        //for rounding to next higher byte
                                 //binary tiff
   if (sampp==1 && bitsps[0]==1) {
                                 // Strip byte counts (ie image size in bytes)
      ifd(0x117,LONG,1,(long)((w+7)/8)*h);
   }
   else {
      for(k=0;k<sampp;k++) bps+=bitsps[k];
      bps>>=3;
                                 // Strip byte counts (ie image size in bytes)
      ifd(0x117,LONG,1,(long)w*h*bps);
   }
   unsigned short x_lut=0;
   long lut_off=0;
   for (k=0;k<sampp;k++) x_lut+=1<<bitsps[k];
   long xresoff=0;
   if (ru != 0) {                // resolution fields
      ifd(0x11a,RATIONAL,1,off); // xres
      xresoff = off;
      off+=8;
      ifd(0x11b,RATIONAL,1,off); // yres
      off+=8;
      ifd(0x128,SHORT,1,ru);     // resolution unit
   }
   if (pal==0 && lut!=NULL) {    // Transfer function
      ifd(0x12d,SHORT,x_lut,off);
      lut_off=off;
      off+=x_lut*2;
   }
   if (dt==NULL) {
      time_t timer;
      struct tm *t;
      timer = time(NULL);
      t = localtime(&timer);
      dt = new char[20];
      if (dt==NULL) error("WriteTiff: could not allocate space for date/time.");
      x=sprintf(&dt[0],"%04d:%02d:%02d %02d:%02d:%02d",
         1900+t->tm_year,t->tm_mon+1,t->tm_mday,t->tm_hour,t->tm_min,t->tm_sec);
      // t->tm_mon returns month starting at January = 0
      // t->tm_year returns the number of years since 1900 (so yr. 2000 is 100)
   }
                                 // Time stamp
   ifd(0x132,ASCII,x=strlen(dt)+1,off);
   long dt_off=off; off+=x;
   long art_off=0;
   if (art != NULL) {
                                 // Name of artist
      ifd(0x13b,ASCII,x=strlen(art)+1,off);
      art_off=off;
      off+=x;
   }
   char str[220];
   long str_off=0;
   long wpl_off=0;
   long chr_off=0;
   if (sampp!=1) {
      if(tiftype==1) {
         sprintf(str,"%lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",
            ms.e[0][0],ms.e[0][1],ms.e[0][2],
            ms.e[1][0],ms.e[1][1],ms.e[1][2],
            ms.e[2][0],ms.e[2][1],ms.e[2][2],
            wp[0],wp[1]);
                                 // Scanner matrix and xy of white point.
         ifd(0x80cf,ASCII,x=strlen(str)+1,off);
         str_off=off; off+=x;
         //      nifd--;
      }
      else if(tiftype==0) {
                                 // white point chromaticity
         ifd(0x13e,RATIONAL,2,off);
         wpl[0][0]=(long) (1000000*chrm[6]);  wpl[0][1]=(long) 1000000;
         wpl[1][0]=(long) (1000000*chrm[7]);  wpl[1][1]=(long) 1000000;
         wpl_off=off; off+=sizeof(wpl);
                                 // primary chromaticities  (in ms)
         ifd(0x13f,RATIONAL,6,off);
         chr[0][0][0]= (long) (1000000*chrm[0]);  chr[0][0][1]= (long) 1000000;
         chr[0][1][0]= (long) (1000000*chrm[1]);  chr[0][1][1]= (long) 1000000;
         chr[1][0][0]= (long) (1000000*chrm[2]);  chr[1][0][1]= (long) 1000000;
         chr[1][1][0]= (long) (1000000*chrm[3]);  chr[1][1][1]= (long) 1000000;
         chr[2][0][0]= (long) (1000000*chrm[4]);  chr[2][0][1]= (long) 1000000;
         chr[2][1][0]= (long) (1000000*chrm[5]);  chr[2][1][1]= (long) 1000000;
         chr_off=off; off+=sizeof(chr);
      }
   }
   // Color Map for palettized images
   long pal_off=off;
   if (pal != 0) {
      ifd(0x140,SHORT,768,off);  // Color Map
      off+=768*2;
   }
   long brite_off=0;
   if (brcosh != NULL) {
      ifd(0x80d1,ASCII,x=48,off);// Image brightness
      brite_off=off; off+=x;
   }
   else if (brite != BRITE) {
      brcosh = new char[48];
      sprintf(brcosh,"B=%lg",brite);
      ifd(0x80d1,ASCII,x=48,off);// Image brightness
      brite_off=off; off+=x;
   }
   long cpyr_off=0;
   if (cpyr != NULL) {
                                 // Copyright
      ifd(0x8298,ASCII,x=strlen(cpyr)+1,off);
      cpyr_off=off;
      off+=x;
   }
   // Terminate list with a null
   len=0;
   s=write(id,&len,4);             // Pointer to non-existent next ifd.
   // Now we write out all the variable fields
   imgoff=off;
   //seek(8) ; write(id,&nifd,2);
   seek(stripoff_loc); s=write(id,&imgoff,4);
   if (sampp>1) {seek(bitsps_off);   s=write(id,bitsps,2*sampp);}
   if (doc!=NULL) {seek(doc_off);    s=write(id,&doc[0],strlen(&doc[0])+1);}
   if (descr != NULL) {
      seek(descr_off);    s=write(id,&descr[0],strlen(&descr[0])+1);
   }
   if (pal==0) {
      if (lut!=NULL) {seek(lut_off);      s=write(id,&lut[0],x_lut*2);}
   }
   else {
      unsigned char cm[2*768];
      unsigned short i,j;
      seek(pal_off);
      memset(cm,0,2*768);
      for (i=1,j=0;i<512;i+=2) {
         cm[i] = palpt->pall[j++];
         cm[512+i] = palpt->pall[j++];
         cm[1024+i] = palpt->pall[j++];
      }
      s=write(id,cm,2*768);
   }
   if (ru != 0) {
      unsigned long res[2];
                                 // Trunc
      res[0] = (unsigned long) (xres*100 + .5);
      res[1] = 100;
      seek(xresoff);
      s=write(id,&res[0],8);
                                 // Trunc
      res[0] = (unsigned long) (yres*100 + .5);
      s=write(id,res,8);
   }
   seek(dt_off);       s=write(id,dt,strlen(dt)+1);
   if (art != NULL) {seek(art_off);      s=write(id,art,strlen(art)+1); }
   if (sampp!=1) {
     if(tiftype==1) {
       seek(str_off);
       s=write(id,str,strlen(str)+1);
     }
     else if(tiftype==0) {
       seek(wpl_off);      s=write(id,&wpl[0][0],sizeof(wpl));
       seek(chr_off);      s=write(id,&chr[0][0][0],sizeof(chr));
     }
   }
   if (brcosh != NULL) {
      seek(brite_off);      s=write(id,brcosh,48);
   }
   if (cpyr != NULL) {seek(cpyr_off);      s=write(id,cpyr,strlen(cpyr)+1);}
   seek(imgoff);
   //  delete[] strbuf;
}


void PCXImage::WriteHeader(char* fname)
{
   struct
   {
      char manuf,
         ver  ,
         enc  ,
         bpp  ;
      short xmin,ymin,xmax,ymax,
         hres, vres;
      char clrmap[48],
         xxx,
         nplanes;
      short bpl,palinfo;
      char filler[58];
   } head;

   head.manuf = 10;
   head.ver=5;
   head.enc=1;
   head.bpp=8;
   head.xmin=0;
   head.ymin=0;
   head.hres=1024;
   head.vres=768;
   head.nplanes=1;
   head.palinfo=1;

   //open output file
   id = open(fname,O_CREAT|O_WRONLY|O_TRUNC|O_BINARY,S_IWRITE);
   if (id<0) error("Cannot open file %s.\n",fname);
   //write output header
   head.xmax = w-1;
   head.ymax = h-1;
   if (w%2 == 1) odd = 1;
   head.bpl = w+odd;
   size_t s=write(id,&head,128);
}


void PCXImage::OpenLine(int num)
{
   //header read/write is complete, file pointer is at beginning of data
   //cannot handle read and write, just one or the other.
   //set up buffer
   nw=w*sizepx;
   mem = num;
   nbuf = BUFSIZE(this);
   amt = (long)(nbuf+1)*nw;
   lbuf = new unsigned char[amt];
   if (lbuf==NULL) error("Could not allocate space for read/write buffer.");
   amt -= nw;                    // space for uncompressed data
   tbuf = &lbuf[0]+amt;
   imgoff = lseek(id,0,SEEK_CUR);
   cur = -1;
   bamt = 0;
}


void* PCXImage::PutLine(int num)
{
   int i,n,k;
   unsigned char *obuf;
   if (num != -1) error("Can't handle non-sequential output.\n");
   if (cur == -1) {
      cur++;
      return &tbuf[0];
   }
                                 //write out compressed data
   if (amt-bamt < (unsigned long)(2*nw)) {
      size_t s=write(id,&lbuf[0],bamt);
      bamt = 0;
   }
   // compress output line at tbuf and put in lbuf
   k=0;i=0;
   obuf = &lbuf[bamt];
   while(i<w) {
      n = 1;
      while((tbuf[i]==tbuf[i+1]) && (i<w-1) && (n<63)) {
         n++;
         i++;
      }
      if(((tbuf[i]&0xc0) == 0xc0) || (n>1)) {
         obuf[k++] = 0xc0 | n;
         obuf[k++] = tbuf[i++];
      }
      else
         obuf[k++] = tbuf[i++];
   }
   if(odd>0) obuf[k++]=0;
   bamt += k;
   return &tbuf[0];
}


void PCXImage::CloseLine()
{
   int i,n,k;
   unsigned char *obuf;
   if (cur==0) {                 //file was opened for writing
                                 //write out compressed data
      if (amt-bamt < (unsigned long)(2*nw)) {
         size_t s=write(id,&lbuf[0],bamt);
         bamt = 0;
      }
      // compress output line (last in lbuf) and put in lbuf
      k=0;i=0;
      obuf = &lbuf[bamt];
      while(i<w) {
         n = 1;
         while((tbuf[i]==tbuf[i+1]) && (i<w-1) && (n<63)) {
            n++;
            i++;
         }
         if(((tbuf[i]&0xc0) == 0xc0) || (n>1)) {
            obuf[k++] = 0xc0 | n;
            obuf[k++] = tbuf[i++];
         }
         else
            obuf[k++] = tbuf[i++];
      }
      if(odd>0) obuf[k++]=0;
      bamt += k;
      size_t s=write(id,&lbuf[0],bamt);
      //write palette
      n=12;
      s=write(id,&n,1);
      if (palpt == NULL) error("Palette for image has not been defined.");
      s=write(id,palpt->pall,768);
   }
   delete[] lbuf;
}


void BMPImage::ReadHeader(char* fname)
{
   char ptbl[1024];
   struct
   {
      short a1;
      long fsize,zero,hsize,hedlen;
   }hda;
   struct hdos2
   {
      short w,h,plane,bitct;
   };
   struct hdwin
   {
      long w,h;
      short plane,bitct;
      long comp,isize,xppm,yppm,clru,clrimp;
   };
   short bitct=0;
   //open input file
   id = open(fname,O_RDONLY|O_BINARY);
   if (id<0) error("Cannot open file %s.\n",fname);
   //read input header
   size_t s=read(id,&hda,18);
   if (hda.a1 != 0x4d42) error("Input file is not BMP.\n");
   if (hda.hedlen==12) {         // OS/2 header
      bmptype = 0;
      hdos2 hd;
      s=read(id,&hd,8);
      bitct = hd.bitct;
      sizepx = bitct/8;
      w = hd.w;
      h = hd.h;
   }
   else if (hda.hedlen==40) {    // windows header
      bmptype = 1;
      hdwin hd;
      s=read(id,&hd,36);
      bitct = hd.bitct;
      sizepx = bitct/8;
      w = hd.w;
      h = hd.h;
   }
   else error("Invalid BMP header length field.\n");
   if (sizepx == 1) {
      short i,j,k;
      if (bmptype ==0) {
         k=3;
         s=read(id,ptbl,768);
      }
      else {
         k=4;
         s=read(id,ptbl,1024);
      }
      palpt = new Pal;
      // rearrange from BMP(BGR) to pall(RGB)
      for (i=0,j=0;i<768;i+=3,j+=k) {
         palpt->pall[i+2] = ptbl[j];
         palpt->pall[i+1] = ptbl[j+1];
         palpt->pall[i] = ptbl[j+2];
      }
   }
   else if (sizepx ==3) {
      sampp = 3;
      bitsps[1] = 8;
      bitsps[2] = 8;
   }
   else if (bitct == 4) {
      bitsps[0] = bitct;
      short i,j,k;
      if (bmptype ==0) {
         k=3;
         s=read(id,ptbl,48);
      }
      else {
         k=4;
         s=read(id,ptbl,64);
      }
      palpt = new Pal;
      // rearrange from BMP(BGR) to pall(RGB)
      for (i=0,j=0;i<48;i+=3,j+=k) {
         palpt->pall[i+2] = ptbl[j];
         palpt->pall[i+1] = ptbl[j+1];
         palpt->pall[i] = ptbl[j+2];
      }
   }
   else if (bitct == 1) {
      bitsps[0] = 1;
      sampp = 1;
      short i,j,k;
      if (bmptype ==0) {
         k=3;
         s=read(id,ptbl,6);
      }
      else {
         k=4;
         s=read(id,ptbl,8);
      }
      palpt = new Pal;
      // rearrange from BMP(BGR) to pall(RGB)
      for (i=0,j=0;i<6;i+=3,j+=k) {
         palpt->pall[i+2] = ptbl[j];
         palpt->pall[i+1] = ptbl[j+1];
         palpt->pall[i] = ptbl[j+2];
      }
      if (ptbl[0] == 0) pi = 1;
      else pi = 0;
   }
   else
      error("Cannot read BMP file with a bit count of %d.\n",bitct);
   if (sizepx > 0)
      nw = sizepx*w;
   else if (bitsps[0]==1)
      nw = (w+7)/8;
   else
      nw = (w<<3)/bitct;
   odd = (4-nw%4)%4;
   nw += odd;
}


void BMPImage::WriteHeader(char* fname)
{
   struct hdos2
   {
      short a1;
      long fsize,zero,hsize,hedlen;
      short w,h,plane,bitct;
   };
   struct hdwin
   {
      short a1;
      long fsize,zero,hsize,hedlen,w,h;
      short plane,bitct;
      long comp,isize,xppm,yppm,clru,clrimp;
   };
   int palsize;
   if (sizepx > 0)
      nw = sizepx*w;
   else {
      nw = (w+7)/8;
      bmptype = 1;               //force Windows BMP format
   }
   odd = (4-nw%4)%4;
   nw += odd;
   if (bmptype == 0) {           // OS/2
      char ptbl[768];
      palsize = 768;
      hdos2 hd;
      hd.w= w;
      hd.h = h;
      hd.a1 = 0x4d42;
      hd.hedlen = 12;
      hd.zero = 0;
      hd.plane = 1;
      hd.hsize = 14+hd.hedlen;
      if (sizepx == 3)
         hd.bitct = 24;
      else if (sizepx ==1) {
         hd.hsize = hd.hsize+768;
         hd.bitct = 8;

         short i;
         if (palpt == NULL) error("Palette for image has not been defined.");
         // rearrange from pall(RGB) to BMP(BGR)
         for (i=0;i<768;i+=3) {
            ptbl[i]   = palpt->pall[i+2];
            ptbl[i+1] = palpt->pall[i+1];
            ptbl[i+2] = palpt->pall[i];
         }
      }
      else
         error("Cannot write BMP file with pixel size other than 1 or 3.\n");
      hd.fsize = hd.hsize+(long)nw*h;
      //open output file
      id = open(fname,O_CREAT|O_WRONLY|O_TRUNC|O_BINARY,S_IWRITE);
      if (id<0) error("Cannot open file %s.\n",fname);
      //write output header
      size_t s=write(id,&hd,sizeof(hd));
      if (sizepx == 1) s=write(id,ptbl,768);
   }
   else {                        //windows bmp file
      char ptbl[1024];
      palsize = 1024;
      hdwin hd;
      hd.w= w;
      hd.h = h;
      hd.a1 = 0x4d42;
      hd.hedlen = 40;
      hd.zero = 0;
      hd.plane = 1;
      hd.hsize = 14+hd.hedlen;
      if (sizepx == 3)
         hd.bitct = 24;
      else if (sizepx ==1) {
         hd.hsize = hd.hsize+1024;
         hd.bitct = 8;
         hd.clru = 256;
         hd.clrimp = 256;
         memset(ptbl,0,1024);

         short i,j;
         if (palpt == NULL) error("Palette for image has not been defined.");
         // rearrange from pall(RGB) to BMP(BGR)
         for (i=0,j=0 ;i<1024;i+=4,j+=3) {
            ptbl[i]   = palpt->pall[j+2];
            ptbl[i+1] = palpt->pall[j+1];
            ptbl[i+2] = palpt->pall[j];
         }
      }
      else if (sampp==1 && bitsps[0]==1) {
         hd.hsize = hd.hsize+8;
         hd.bitct = 1;
         hd.clru = 0;
         hd.clrimp = 0;
         palsize = 8;
         memset(ptbl,0,8);
         if (pi == 0)
            ptbl[0] = ptbl[1] = ptbl[2] = 255;
         else
            ptbl[4] = ptbl[5] = ptbl[6] = 255;
      }
      else
         error("Cannot write BMP file with pixel size other than 1, 8 or 24 bits.\n");
      hd.comp = 0;
      hd.isize = (long)nw*h;
      hd.fsize = hd.hsize+hd.isize;
      hd.xppm = 0;               // xres*100./2.54; //pixels per meter
      hd.yppm = 0;               // yres*100./2.54;
      //open output file
      id = open(fname,O_CREAT|O_WRONLY|O_TRUNC|O_BINARY,S_IWRITE);
      if (id<0) error("Cannot open file %s.\n",fname);
      //write output header
      size_t s=write(id,&hd,sizeof(hd));
      if (sizepx <= 1) s=write(id,ptbl,palsize);
   }
}


void BMPImage::OpenLine(int num)
{
   //header read/write is complete, file pointer is at beginning of data
   //cannot handle read and write, just one or the other.
   //set up buffer
   mem = num;
   nbuf = BUFSIZE(this);
   amt = (long)(nbuf+1)*nw;
   lbuf = new unsigned char[amt];
   if (lbuf==NULL) error("Could not allocate space for read/write buffer.");
   memset(&lbuf[0],0,amt);
   amt -= nw;
   tbuf = &lbuf[0]+amt;
   first = 0;
   last = -1;
   imgoff = lseek(id,0,SEEK_CUR);
   imgend = imgoff + (long)nw*h;
   cur = -1;
}


void* BMPImage::PutLine(int num)
{
   int n,m;
   cur = (num < 0 ) ? cur+1 : num ;
   if(cur >= first+nbuf) {       // need to write data
      //reverse order of rows
      n = (cur-first-1)*nw;
      for (m=0;m<n;m+=nw,n-=nw) {
         memcpy(tbuf,&lbuf[n],nw);
         memcpy(&lbuf[n],&lbuf[m],nw);
         memcpy(&lbuf[m],tbuf,nw);
      }
      //write out data
      n=lseek(id,imgoff+(long)(h-cur)*nw,SEEK_SET);
      if (sizepx == 3) {
         //rearrange data in lbuf from RGB to BGR by row because of padding
         int i,j,ix;
         unsigned char *tbf,t;
         tbf = &lbuf[0];
         ix = cur-first;
         for (i=0;i<ix;i++) {
            for (j=0;j<nw-odd;j+=3) {
               t = tbf[j];
               tbf[j] = tbf[j+2];
               tbf[j+2] = t;
            }
            tbf += nw;
         }
      }
      size_t s=write(id,&lbuf[0],(long)(cur-first)*nw);
      first = cur;
   }
   return &lbuf[(long)(cur-first)*nw];
}


void* BMPImage::GetLine(int num)
{
   int m;
   int n,ix;
   cur = (num < 0) ? cur+1 : num;
   if (cur>last || cur<first) {  // need to get more data
      m = h-(cur+nbuf);
      n = nbuf;
      if (m<0) {
         n = nbuf+m;
         m = 0;
      }
      first = cur;
      last = cur+(n-1);
      ix = n;
      n *= nw;
      lseek(id,imgoff+(long)m*nw,SEEK_SET);
      size_t s=read(id,lbuf,n);
      n -= nw;
      //reverse order of rows
      for (m=0;m<n;m+=nw,n-=nw) {
         memcpy(tbuf,&lbuf[n],nw);
         memcpy(&lbuf[n],&lbuf[m],nw);
         memcpy(&lbuf[m],tbuf,nw);
      }
      if (sizepx == 3) {
         //rearrange data in lbuf from RGB to BGR by row because of padding
         int i,j;
         unsigned char *tbf,t;
         tbf = &lbuf[0];
         for (i=0;i<ix;i++) {
            for (j=0;j<nw-odd;j+=3) {
               t = tbf[j];
               tbf[j] = tbf[j+2];
               tbf[j+2] = t;
            }
            tbf += nw;
         }
      }
   }
   return &lbuf[(long)(cur-first)*nw];
}


void BMPImage::CloseLine()
{
   int n,m;
   if((cur >= first) && (last < 0)) {
   // need to write data --always  if generating a BMP file
      //reverse order of rows
      n = (cur-first)*nw;
      for (m=0;m<n;m+=nw,n-=nw) {
         memcpy(tbuf,&lbuf[n],nw);
         memcpy(&lbuf[n],&lbuf[m],nw);
         memcpy(&lbuf[m],tbuf,nw);
      }
      //write out data
      if (h-1-cur != 0) error("Did not write correct amount of data.");
      n=lseek(id,imgoff,SEEK_SET);
      if (sizepx == 3) {
         //rearrange data in lbuf from RGB to BGR by row because of padding
         int i,j,ix;
         unsigned char *tbf,t;
         tbf = &lbuf[0];
         ix = cur+1-first;
         for (i=0;i<ix;i++) {
            for (j=0;j<nw-odd;j+=3) {
               t = tbf[j];
               tbf[j] = tbf[j+2];
               tbf[j+2] = t;
            }
            tbf += nw;
         }
      }
      size_t s=write(id,&lbuf[0],(long)(cur+1-first)*nw);
   }
   delete[] lbuf;
}
