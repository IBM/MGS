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

#ifndef _lenslexer_h
#define _lenslexer_h
#include "Copyright.h"

#include <FlexLexer.h>
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
