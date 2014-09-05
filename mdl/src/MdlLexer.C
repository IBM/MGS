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

#include "MdlLexer.h"
#include <iostream>
#include <sstream>

MdlLexer::MdlLexer(std::istream * infile, std::ostream * outfile)
: yyFlexLexer(infile,outfile), yylval(0),lineCount(0)
{
}


MdlLexer::~MdlLexer()
{
}


void MdlLexer::skip_proc(void)
{
   // put input into a std::string
   std::string buffer;
   std::ostringstream ostr;
   int c;
   while ((c=yyinput())!= '\n') {
      ostr <<char(c);
   }
   buffer = ostr.str();


   // grab required values and compute lineOffset
   std::istringstream istr(buffer);
   istr >> lineCount;
   istr >> currentFileName;
}

const char* MdlLexer::getToken() 
{
   return yytext;
}
