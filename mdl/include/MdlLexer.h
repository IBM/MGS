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

#ifndef MdlLexer_H
#define MdlLexer_H
#include "Mdl.h"

#if !defined(yyFlexLexerOnce) 
#include <FlexLexer.h>
#endif 
#include "ParserClasses.h"
#include <string>
#ifndef YYSTYPE_DEFINITION
#define YYSTYPE_DEFINITION
#include "mdl.tab.h"
#endif

class MdlContext;

class MdlLexer : public yyFlexLexer
{
   public:
      MdlLexer(std::istream * infile, std::ostream * outfile);
      ~MdlLexer();

      int lex(YYSTYPE *lvaluep, YYLTYPE *locp, MdlContext *context);
      int yylex();
      void skip_proc(void);

      YYSTYPE *yylval;
      YYLTYPE *yylloc;
      MdlContext *context;
      std::string currentFileName;
      std::string gcppLine;
      int lineCount;
      const char* getToken();
};

inline int MdlLexer::lex(YYSTYPE *lvaluep, YYLTYPE *locp, MdlContext *c)
{
   yylval = lvaluep;
   yylloc = locp;
   context = c;
   return yylex();
}
#endif
