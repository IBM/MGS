// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "GslLexer.h"
//#include <stdio.h>
#include <iostream>
#include <sstream>

GslLexer::GslLexer(std::istream * infile, std::ostream * outfile)
: yyFlexLexer(infile,outfile), yylval(0),lineCount(0)
{
}


GslLexer::~GslLexer()
{
}


void GslLexer::skip_proc(void)
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

const char* GslLexer::getToken() 
{
   return yytext;
}
