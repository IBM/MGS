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

#include "Ini.h"
#include <sys/stat.h>

//Constructor
IniFile::IniFile(void)
{
   char* env;
   inibuf = NULL;
   inipath[0]=0;
   env = getenv("INIPATH");   
   if (env!=NULL) strconcat(inipath,env,"\\");
}


IniFile::IniFile(char* env)
{
   inibuf = NULL;
   inipath[0]=0;
   if (env[0] != 0) strconcat(inipath,env,"\\");
}


//Destructor
IniFile::~IniFile()
{
   delete[] inibuf;
}


int IniFile::IniOpenRd(const char* ifn)
{
   char name[80];
   strcpy(name,inipath);
   strcat(name,ifn);
   id = open(name,O_RDONLY|O_BINARY);
   if (id<0) error("%s ini file not found.\n",name);

   //  lmax = filelength(id);
   // This is Windows code, and uses the filelength function of io.h in MSVC++
   // Replace "filelength(id)" with "stat_buf.st_size"
   // as follows, Ravi Rao, 8/28/02
   struct stat stat_buf;
   fstat(id, &stat_buf);
   lmax = stat_buf.st_size;

   inibuf = new char[lmax];
   if (inibuf == NULL) error("Cannot allocate enough memory for ini file.");
   size_t s=read(id,&inibuf[0],lmax);
   close(id);
   rw = 0;
   return 0;
}


int IniFile::IniOpenRW(char* ifn)
{
   char name[80];
   strcpy(name,inipath);
   strcat(name,ifn);
   id = open(name,O_RDWR|O_BINARY);
   if (id<0) error("Ini file not found.");

   // lmax = filelength(id);
   // This is Windows code, and uses the filelength function of io.h in MSVC++
   // Replace "filelength(id)" with "stat_buf.st_size"
   // as follows, Ravi Rao, 8/28/02
   struct stat stat_buf;
   fstat(id, &stat_buf);
   lmax = stat_buf.st_size;

   xmax = lmax +5000;
   inibuf = new char[xmax];
   if (inibuf == NULL) error("Cannot allocate enough memory for ini file.");
   size_t s=read(id,&inibuf[0],lmax);
   rw = 1;
   return 0;
}


char* IniFile::IniInsert(char* ins)
{
   long n;
   if (rw != 1) error("Ini file not opened for writing.");
   n = strlen(ins);
   if (lmax+n > xmax) return NULL;
   memmove(locptr+n,locptr,lmax-(locptr-&inibuf[0]));
   memcpy(locptr,ins,n);
   lmax += n;
   locptr += n;
   return locptr;
}


char* IniFile::IniInsert(double* v, unsigned short num)
{
   int i,j,k;
   char buf[2000];
   // create a std::string containing values
   k = 0;
   for(i=0;i<num;i+=12) {
      for(j=0;j<12;j+=2) {
         if (i+j >= num) break;
         k += sprintf(buf+k,"%.0f, %.3f, ",v[i+j],v[i+j+1]);
      }
      if (i+j < num)
         k += sprintf(buf+k,"\n  ");
      else
         k += sprintf(buf+k-2,"\n") -1;
   }
   buf[k] = 0;
   return IniInsert(buf);
}


char* IniFile::IniDelete()
{
   int m,n,k;
   char* temptr;
   if (rw != 1) error("Ini file not opened for writing.");
   line[0]=0;
   sscanf(locptr,"%[^\r]",line);
   // find out if line continues on next line
   temptr = 2+strstr(locptr,"\r");
   m = strspn(temptr," .,0123456789");
   k = 0;
   while (m > 0) {
      strncat(line,temptr,m);
      temptr += 2+m;
      k += 2;
      m = strspn(temptr," .,0123456789");
   }
   n = 2+k+strlen(line);
   memmove(locptr,locptr+n,lmax-n-(locptr-&inibuf[0]));
   lmax -= n;
   return locptr;
}


void IniFile::IniClose()
{
   if(rw == 1) {

      // chsize(id,lmax);
      // This is Windows code, and uses the "chsize" function in MSVC++
      // Replace "chsize" with "ftruncate"
      // as follows, Ravi Rao, 8/28/02
      int f=ftruncate(id,lmax);

      lseek(id,0,SEEK_SET);
      size_t s=write(id,&inibuf[0],lmax);
      close(id);
   }
}


char* IniFile::IniFindApp(const char* app)
{
   line[0]=0;
   appptr = locptr = find(&inibuf[0],strconcat(line,"[",app,"]"),lmax,&inibuf[0]);
   return locptr;
}


char* IniFile::IniFindKey(const char* key)
{
   char* temptr;
   line[0]=0;
   locptr = appptr+1;
   temptr = find(locptr,"[",lmax,&inibuf[0]);
   if(temptr==NULL)
      locptr = find(locptr,strconcat(line,key,"="),lmax,&inibuf[0]);
   else
      locptr = find(locptr,strconcat(line,key,"="),temptr-&inibuf[0]+1,&inibuf[0]);
   if (locptr == NULL) {
      locptr = appptr;
      return NULL;
   }
   locptr += strlen(line);
   return locptr;
}


char* IniFile::IniNxtApp()
{
   char* temptr;
   temptr = find(appptr+1,"[",lmax,&inibuf[0]);
   if(temptr==NULL)
      temptr = lmax+&inibuf[0];
   return temptr;
}


char* IniFile::IniGetNxt()
{
   line[0]=0;
   sscanf(locptr,"%[^\r]",line);
   locptr += 2+strlen(line);
   line[0]=0;
   sscanf(locptr,"%[^\r]",line);
   if (locptr > inibuf+lmax-2) return NULL;
   else return line;
}


char* IniFile::IniGetCmt()
{
   char* temptr;
   if (locptr == appptr)
      temptr = find(locptr+2,"[",lmax,&inibuf[0]);
   else
      temptr = find(locptr,"[",lmax,&inibuf[0]);
   if(temptr==NULL)
      locptr = find(locptr,";",lmax,&inibuf[0]);
   else
      locptr = find(locptr,";",temptr-&inibuf[0]+1,&inibuf[0]);
   if (locptr == NULL) {
      locptr = appptr;
      return NULL;
   }
   return locptr;
}


int IniFile::IniGetStr(char* strsv)
{
   sscanf(locptr,"%s",strsv);
   return strlen(strsv);
}


int IniFile::IniGetFld(char* strsv)
{
   int n = strcspn(locptr,"\r\n");
   strncpy(strsv,locptr,n);
   strsv[n] = 0;
   return strlen(strsv);
}


int IniFile::IniGetVal(double* v, unsigned short num)
{
   short n,m;
   char* temptr;
   sscanf(locptr,"%[^\r]",line);
   // find out if line continues on next line
   temptr = 2+strstr(locptr,"\r");
   m = strspn(temptr," .,0123456789");
   while (m > 0) {
      strncat(line,temptr,m);
      temptr += 2+m;
      m = strspn(temptr," .,0123456789");
   }
   temptr = strtok(line," ,");
   for (n=0;n<num;n++) {
      if (temptr==NULL) break;
      v[n] = atof(temptr);
      temptr = strtok(NULL," ,\r");
   }
   return n;
}


int IniFile::IniGetVal(short* v, unsigned short num)
{
   short n,m;
   char* temptr;
   sscanf(locptr,"%[^\r]",line);
   // find out if line continues on next line
   temptr = 2+strstr(locptr,"\r");
   m = strspn(temptr," .,0123456789");
   while (m > 0) {
      strncat(line,temptr,m);
      temptr += 2+m;
      m = strspn(temptr," .,0123456789");
   }
   temptr = strtok(line," ,");
   for (n=0;n<num;n++) {
      if (temptr==NULL) break;
      v[n] = atoi(temptr);
      temptr = strtok(NULL," ,\r");
   }
   return n;
}


int IniFile::IniGetVal(short& v)
{
   char* temptr;
   sscanf(locptr,"%[^\r]",line);
   temptr = strtok(line," ,");
   if (temptr==NULL) return 0;
   v = atoi(temptr);
   return 1;
}


int IniFile::IniGetVal(double& v)
{
   char* temptr;
   sscanf(locptr,"%[^\r]",line);
   temptr = strtok(line," ,");
   if (temptr==NULL) return 0;
   v = atof(temptr);
   return 1;
}


char* find(char *p, const char *s, long lmax, char *inibuf)
{
   while(p<inibuf+lmax) {
      p=strstr(p,s);
      if(p!=NULL && p<inibuf+lmax)
         if(p[-1]=='\n' || p==inibuf) return p;
      else p += strlen(s);
      else return NULL;
   }
   return NULL;
}
