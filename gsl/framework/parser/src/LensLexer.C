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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "LensLexer.h"
//#include <stdio.h>
#include <iostream>
#include <sstream>

LensLexer::LensLexer(std::istream * infile, std::ostream * outfile)
: yyFlexLexer(infile,outfile), yylval(0),lineCount(0)
{
}


LensLexer::~LensLexer()
{
}


void LensLexer::skip_proc(void)
{
   // put input into a std::string
   std::string buffer;
   std::ostringstream ostr;
   int c;
   while ((c=yyinput())!= '\n') {
      ostr <<char(c);
   }
   buffer = ostr.str();

   // Provide some feedback
   // std::cerr <<buffer <<std::endl;
   // std::cerr << "PREPROC: i = "<<buffer.size()<<std::endl;
   // std::cerr << "PREPROC: "<<buffer<<std::endl;

   // grab required values and compute lineOffset
   std::istringstream istr(buffer);
   istr >> lineCount;
   istr >> currentFileName;

   // provide additional feedback
   // std::cerr << "PREPROC: line "<<gcppLineNumber<<", file "<< currentFileName <<std::endl;
   // std::cerr <<"PREPROC: line offset = "<<lineOffset<<std::endl;

}

const char* LensLexer::getToken() 
{
   return yytext;
}
