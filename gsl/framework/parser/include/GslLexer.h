// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GSLLEXER_H
#define GSLLEXER_H
#include "Copyright.h"

#if !defined(yyFlexLexerOnce) 
#include <FlexLexer.h>
#endif 
#include "GslParser.h"
#include <string>
#ifndef YYSTYPE_DEFINITION
#define YYSTYPE_DEFINITION
#include "speclang.tab.h"
#endif

class GslContext;

class GslLexer : public yyFlexLexer
{
   public:
      GslLexer(std::istream * infile, std::ostream * outfile);
      ~GslLexer();

      int lex(YYSTYPE *lvaluep, YYLTYPE *locp, GslContext *context);
      int yylex();
      void skip_proc(void);

      const char* getToken();

      YYSTYPE *yylval;
      YYLTYPE *yylloc;
      GslContext *context;
      std::string currentFileName;
      std::string gcppLine;
      int lineCount;
};

inline int GslLexer::lex(YYSTYPE *lvaluep, YYLTYPE *locp, GslContext *c)
{
   yylval = lvaluep;
   yylloc = locp;
   context = c;
   return yylex();
}
#endif
