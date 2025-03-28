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

#ifndef YYSTYPE
#define YYSTYPE_IS_DECLARED 1
#include "mdl_types.h" 

// This must match exactly what's in mdl.tab.h
union YYSTYPE {
   double V_double;
   int V_int;
   Connection::ComponentType V_connectionComponentType;
   std::string *P_string;
   C_array *P_array;
   C_connection *P_connection;
   C_constant *P_constant;
   C_dataType *P_dataType;
   C_dataTypeList *P_dataTypeList;
   C_edge *P_edge;
   C_functor *P_functor;
   C_general *P_general;
   C_generalList *P_generalList;
   C_identifierList *P_identifierList;
   C_interface *P_interface;
   C_interfacePointer *P_interfacePointer;
   C_interfacePointerList *P_interfacePointerList;
   C_interfaceMapping *P_interfaceMapping;
   C_instanceMapping *P_instanceMapping;
   C_nameComment *P_nameComment;
   C_nameCommentList *P_nameCommentList;
   C_node *P_node;
   C_noop *P_noop;
   C_phase *P_phase;
   C_phaseIdentifier *P_phaseIdentifier;
   C_phaseIdentifierList *P_phaseIdentifierList;
   C_predicateFunction *P_predicateFunction;
   C_psetMapping *P_psetMapping;
   C_returnType *P_returnType;
   C_shared *P_shared;
   C_sharedMapping *P_sharedMapping;
   C_struct *P_struct;
   C_triggeredFunction *P_triggeredFunction;
   C_typeClassifier *P_typeClassifier;
   C_typeCore *P_typeCore;
   C_userFunction *P_userFunction;
   C_userFunctionCall *P_userFunctionCall;
   C_variable *P_variable;
   Predicate *P_predicate;
};
#endif

#include "Mdl.h"
#include "bison_compat.h"  // Include compatibility layer first
#include "mdl.tab.h"
#include "ParserClasses.h"

#if !defined(yyFlexLexerOnce)
#include <FlexLexer.h>
#endif 

#include <string>

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
      void debugLocation() {
      fprintf(stderr, "DEBUG: Current location is %s:%d\n", 
         currentFileName.c_str(), lineCount);
      }
};

inline int MdlLexer::lex(YYSTYPE *lvaluep, YYLTYPE *locp, MdlContext *c)
{
   yylval = lvaluep;
   yylloc = locp;
   context = c;
   int result = yylex();
   // Optional debug: if (result == DOUBLE || result == INT) debugLocation();
   return result;
}
#endif
