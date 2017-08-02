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

/* contains class for reading ini files.
   FindKey points to data after identifying std::string
#include "Copyright.h"
   Other Finds points to beginning of field
   Gets convert data to specified form
*/

#ifndef INI_H
#define INI_H

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// Inserted, Ravi Rao, 8/28/02
// as this could not be found in any of the system .h
// files on AIX, and this is required by the code
// (ported from Windows).
#ifndef O_BINARY
#define O_BINARY 0
#endif

#include "ImgUtil.h"
class IniFile
{
   int id;
   int rw;
   long lmax,xmax;
   char line[1200];              //found std::string
   char *inibuf, *locptr, *appptr;
   char inipath[80];
   public:
      // constructor
      IniFile(void);
      IniFile(char* pathx);
      // functions
      ~IniFile();
      char* IniGetPath(void){return inipath;};
      int IniOpenRd(const char* ifn);
      int IniOpenRW(char* ifn);
      char* IniInsert(char* ins);
      char* IniInsert(double* v, unsigned short num);
      char* IniDelete();
      void IniClose();
      char* IniFindApp(const char* app);
      char* IniFindKey(const char* key);
      char* IniNxtApp();
      char* IniGetNxt();
      char* IniGetCmt();
      int IniGetStr(char* strsv);//gets first token as std::string
      int IniGetFld(char* strsv);//gets entire line after =
      int IniGetVal(double* v, unsigned short num);
      int IniGetVal(short* v, unsigned short num);
      int IniGetVal(short& v);
      int IniGetVal(double& v);
};
char* find(char* buf, const char* str, long lmax, char* inibuf);
#endif
