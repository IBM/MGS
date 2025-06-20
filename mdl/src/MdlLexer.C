// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "MdlLexer.h"
#include <iostream>
#include <sstream>

MdlLexer::MdlLexer(std::istream * infile, std::ostream * outfile)
: yyFlexLexer(infile,outfile), yylval(0),lineCount(0) {
}

MdlLexer::~MdlLexer() {
}

void MdlLexer::skip_proc(void) {
   assert(0);
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
   // Optional debug: debugLocation();
}

const char* MdlLexer::getToken() {
   return yytext;
}
