// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _lenslexer_h
#define _lenslexer_h
#include "Copyright.h"

#if !defined(yyFlexLexerOnce) 
#include <FlexLexer.h>
#endif 
#include "LensParser.h"
#include <string>
#ifndef YYSTYPE_DEFINITION
#define YYSTYPE_DEFINITION
#include "speclang.tab.h"
#endif

class LensContext;

class LensLexer : public yyFlexLexer
{
   public:
      LensLexer(std::istream * infile, std::ostream * outfile);
      ~LensLexer();

      int lex(YYSTYPE *lvaluep, YYLTYPE *locp, LensContext *context);
      int yylex();
      void skip_proc(void);

      const char* getToken();

      YYSTYPE *yylval;
      YYLTYPE *yylloc;
      LensContext *context;
      std::string currentFileName;
      std::string gcppLine;
      int lineCount;
};

inline int LensLexer::lex(YYSTYPE *lvaluep, YYLTYPE *locp, LensContext *c)
{
   yylval = lvaluep;
   yylloc = locp;
   context = c;
   return yylex();
}
#endif
