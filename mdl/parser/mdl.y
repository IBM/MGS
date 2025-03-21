%{
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
%}

%{
#include "MdlLexer.h"
#include "ParserClasses.h"
#include "MdlContext.h"
#include "Initializer.h"

#include "StringType.h"
#include "BoolType.h"
#include "CharType.h"
#include "ShortType.h"
#include "IntType.h"
#include "LongType.h"
#include "FloatType.h"
#include "DoubleType.h"
#include "LongDoubleType.h"
#include "UnsignedType.h"
#include "EdgeType.h"
#include "EdgeSetType.h"
#include "EdgeTypeType.h"
#include "FunctorType.h"
#include "GridType.h"
#include "NodeType.h"
#include "NodeSetType.h"
#include "NodeTypeType.h"
#include "ServiceType.h"
#include "RepertoireType.h"
#include "TriggerType.h"
#include "ParameterSetType.h"
#include "NDPairListType.h"

#include "Operation.h"
#include "ParanthesisOp.h"
#include "TerminalOp.h"
#include "InFixOp.h"
#include "AllValidOp.h"
#include "EqualOp.h"
#include "NotEqualOp.h"
#include "GSValidOp.h"
#include "LessEqualOp.h"
#include "GreaterEqualOp.h"
#include "LessOp.h"
#include "GreaterOp.h"
#include "BValidOp.h"
#include "AndOp.h"
#include "OrOp.h"

#include "Connection.h"

#include "PhaseType.h"
#include "PhaseTypeInstance.h"
#include "PhaseTypeShared.h"
#include "PhaseTypeGridLayers.h"

#include "TriggeredFunction.h"

#include "SyntaxErrorException.h"
#include "InternalException.h"
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <stdio.h>
#include <chrono>
#include <time.h>

using namespace std;

#define USABLE
#define YYPARSE_PARAM parm
#define YYLEX_PARAM parm
#ifndef YYDEBUG
#define YYDEBUG 1
#endif
#define CONTEXT ((MdlContext *) parm)
#define yyparse mdlparse
#define yyerror mdlerror
#define yylex   mdllex
#define CURRENTFILE (((MdlContext *) parm)->_lexer)->currentFileName
#define CURRENTLINE (((MdlContext *) parm)->_lexer)->lineCount


   void mdlerror(const char *s);
   void mdlerror(YYLTYPE*, void*, const char *s);
   int mdllex(YYSTYPE *lvalp, YYLTYPE *locp, void *context);

   inline void HIGH_LEVEL_EXECUTE(void* parm, C_production* l) {
      try{
	 l->execute(CONTEXT);
	 delete l;
	 CONTEXT->setErrorDisplayed(false);
      } catch (SyntaxErrorException& e) {
	 cerr << "Error at file:";
	 if (e.isCaught()) {
	    cerr << e.getFileName() << ", line:" << e.getLineNumber() << ", ";
	 } else {
	    MdlLexer *li = CONTEXT->_lexer;
	    cerr << li->currentFileName << ", line:" << li->lineCount << ", ";
	 }
	 cerr << e.getError() << endl; 
	 CONTEXT->setError();
	 delete l;
      } catch (InternalException& e) {
	 cerr << "Error at file:";	 
	 MdlLexer *li = CONTEXT->_lexer;
	 cerr << li->currentFileName << ", line:" << li->lineCount << ", ";
	 cerr << e.getError() << endl; 
	 CONTEXT->setError();
	 delete l;
      } catch (...) {
	 cerr << "Error at file:";	 
	 MdlLexer *li = CONTEXT->_lexer;
	 cerr << li->currentFileName << ", line:" << li->lineCount << ", ";
	 CONTEXT->setError();
	 delete l;
      }

   }
%}

%pure-parser
%locations
%parse-param       {void * YYPARSE_PARAM}
%lex-param       {void * YYLEX_PARAM}
/*%param { void * context} */

%{
#ifndef YYSTYPE_DEFINITION
#define YYSTYPE_DEFINITION
%}


%union {
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
} 

%{
#endif
%}

%token <V_double>  DOUBLE_CONSTANT
%token <V_int> INT_CONSTANT
%token <P_string> STRING_LITERAL
%token <P_string> IDENTIFIER
%token STRING
%token BOOL
%token CHAR
%token SHORT
%token INT
%token LONG
%token FLOAT
%token DOUBLE
%token UNSIGNED
%token EDGE
%token EDGESET
%token EDGETYPE
%token FUNCTOR
%token GRID
%token NODE
%token NODESET
%token NODETYPE
%token SERVICE
%token REPERTOIRE
%token TRIGGER
%token TRIGGEREDFUNCTION
%token SERIAL
%token PARALLEL
%token PARAMETERSET
%token NDPAIRLIST
%token OR
%token AND
%token EQUAL
%token NOT_EQUAL
%token LESS_EQUAL
%token GREATER_EQUAL
%token LESS
%token GREATER
%token DOT
%token AMPERSAND
%token LEFTSHIFT
%token RIGHTSHIFT
%token ELLIPSIS 
%token STAR 
%token _TRUE
%token _FALSE
%token DERIVED
%token STRUCT
%token INTERFACE
%token CONNECTION
%token PRENODE
%token POSTNODE
%token EXPECTS
%token IMPLEMENTS
%token SHARED
%token INATTRPSET
%token OUTATTRPSET
%token PSET
%token INITPHASE
%token RUNTIMEPHASE
%token FINALPHASE
%token LOADPHASE
%token CONSTANT
%token VARIABLE
%token USERFUNCTION
%token PREDICATEFUNCTION
%token INITIALIZE
%token EXECUTE
%token CATEGORY
%token VOID
%token PRE
%token POST
%token GRIDLAYERS
%token THREADS
%token OPTIONAL
%token FRAMEWORK

/* types for non-terminals */
%type <P_general> argumentDataType
%type <P_generalList> argumentDataTypeList
// %type <P_general> argumentMapping
// %type <P_generalList> argumentMappingList
%type <P_array> array
%type <V_connectionComponentType> connectionComponentType
%type <P_connection> connection
%type <P_general> connectionStatement
%type <P_generalList> connectionStatementList
%type <P_constant> constant
%type <P_general> constantStatement
%type <P_generalList> constantStatementList
%type <P_dataType> dataType
%type <P_dataTypeList> dataTypeList
%type <P_edge> edge
%type <P_connection> edgeConnection
%type <V_connectionComponentType> edgeConnectionComponentType
%type <P_general> edgeConnectionStatement
%type <P_generalList> edgeConnectionStatementList
%type <P_phase> edgeInstancePhase
%type <P_general> edgeStatement
%type <P_generalList> edgeStatementList
%type <P_general> execute
%type <P_functor> functor
%type <P_general> functorStatement
%type <P_generalList> functorStatementList
%type <P_general> inAttrPSet
%type <P_general> initialize
%type <P_triggeredFunction> triggeredFunctionInstance
%type <P_identifierList> identifierList
%type <P_identifierList> identifierDotList
%type <P_interface> interface
%type <P_interfacePointer> interfacePointer
%type <P_interfacePointerList> interfacePointerList
%type <P_interfaceMapping> interfaceToMember
%type <P_instanceMapping> instanceMapping
%type <P_nameComment> nameCommentArgument
%type <P_nameCommentList> nameCommentArgumentList
%type <P_nameComment> nameComment
%type <P_nameCommentList> nameCommentList
%type <P_phase> nodeInstancePhase
%type <P_node> node
%type <P_general> nodeStatement
%type <P_generalList> nodeStatementList
%type <P_typeClassifier> nonPointerTypeClassifier
%type <P_noop> noop
%type <P_dataType> optionalDataType
%type <P_general> outAttrPSet
%type <P_shared> shared
%type <P_sharedMapping> sharedMapping
%type <P_struct> struct
%type <P_typeClassifier> typeClassifier
%type <P_typeCore> typeCore
%type <P_phaseIdentifier> phaseIdentifier
%type <P_phaseIdentifierList> phaseIdentifierList
%type <P_predicate> predicate
%type <P_psetMapping> psetToMember
%type <P_returnType> returnType
%type <P_phase> sharedPhase
%type <P_general> sharedStatement
%type <P_generalList> sharedStatementList
%type <P_triggeredFunction> triggeredFunctionShared
%type <P_userFunction> userFunction
%type <P_userFunctionCall> userFunctionCall
%type <P_predicateFunction> predicateFunction
%type <P_variable> variable
%type <P_phase> variableInstancePhase
%type <P_general> variableStatement
%type <P_generalList> variableStatementList

%left OR AND EQUAL NOT_EQUAL LESS_EQUAL GREATER_EQUAL LESS GREATER

%start mdlFile

%%

mdlFile:  parserLineList {

}
;

parserLineList: parserLine {

}
| parserLineList parserLine {

}
;

parserLine: struct {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
| interface {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
| edge {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
| node {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
| variable {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
| constant {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
| functor {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
/*
| error ';' {
   MdlLexer *l = ((MdlContext *) parm)->_lexer;
   cerr<< "Error position: "<<l->currentFileName<<": "
       <<l->lineCount<<endl<<endl;
   CONTEXT->setError();
} 
*/
| error {
   MdlContext *c = (MdlContext *) parm;
   MdlLexer *l = c->_lexer;
   if ((c->isSameErrorLine(l->currentFileName, l->lineCount) == false) 
       && !c->isErrorDisplayed()){
      cerr<< "Error at file:"<<l->currentFileName<<", line:" <<l->lineCount<< ", ";
      cerr<< "unexpected token: " << l->getToken() << endl << endl;
      c->setErrorDisplayed(true);
   }
   c->setLastError(l->currentFileName, l->lineCount);
   CONTEXT->setError();
}
;


/* Directives */

struct: STRUCT IDENTIFIER '{' dataTypeList '}'  {
   $$ = new C_struct(*$2, $4);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
} 
| FRAMEWORK STRUCT IDENTIFIER '{' dataTypeList '}'  {
   $$ = new C_struct(*$3, $5, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $3;
} 
;

interface: INTERFACE IDENTIFIER '{' dataTypeList '}'  {
   $$ = new C_interface(*$2, $4);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
} 
;

edge: EDGE IDENTIFIER IMPLEMENTS interfacePointerList '{' edgeStatementList '}' {
   $$ = new C_edge(*$2, $4, $6);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
}
| EDGE IDENTIFIER '{' edgeStatementList '}' {
   $$ = new C_edge(*$2, 0, $4);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
}
;

edgeStatementList: edgeStatement {
   $$ = new C_generalList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| edgeStatementList edgeStatement {
   $$ = new C_generalList($1, $2);   
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

edgeStatement: noop {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| dataType {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| optionalDataType {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| edgeInstancePhase {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| instanceMapping {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| sharedMapping {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| edgeConnection {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| shared {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| inAttrPSet {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| outAttrPSet {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| userFunction {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| predicateFunction {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| triggeredFunctionInstance {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

node: NODE IDENTIFIER IMPLEMENTS interfacePointerList '{' nodeStatementList '}' {
   $$ = new C_node(*$2, $4, $6);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
}
| NODE IDENTIFIER '{' nodeStatementList '}' {
   $$ = new C_node(*$2, 0, $4);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
}
;

nodeStatementList: nodeStatement {
   $$ = new C_generalList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| nodeStatementList nodeStatement {
   $$ = new C_generalList($1, $2);   
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

nodeStatement: noop {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| dataType {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| optionalDataType {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| nodeInstancePhase {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| instanceMapping {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| sharedMapping {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| connection {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| shared {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| inAttrPSet {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| outAttrPSet {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| userFunction {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| predicateFunction {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| triggeredFunctionInstance {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

noop: ';' {
   $$ = new C_noop();
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

inAttrPSet: INATTRPSET '{' dataTypeList '}'  {
   $$ = new C_inAttrPSet($3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
} 
;

outAttrPSet: OUTATTRPSET '{' dataTypeList '}'  {
   $$ = new C_outAttrPSet($3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

shared: SHARED '{' sharedStatementList '}' {
   $$ = new C_shared($3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| SHARED sharedStatement {
   $$ = new C_shared();
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   $$->setGeneral($2);
}
;

sharedStatementList: sharedStatement {
   $$ = new C_generalList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| sharedStatementList sharedStatement {
   $$ = new C_generalList($1, $2);   
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

sharedStatement: noop {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| dataType {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| sharedPhase {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| triggeredFunctionShared {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| optionalDataType {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

variable: VARIABLE IDENTIFIER IMPLEMENTS interfacePointerList '{' variableStatementList '}' {
   $$ = new C_variable(*$2, $4, $6);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
}
| VARIABLE IDENTIFIER '{' variableStatementList '}' {
   $$ = new C_variable(*$2, 0, $4);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
}
;

variableStatementList: variableStatement {
   $$ = new C_generalList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| variableStatementList variableStatement {
   $$ = new C_generalList($1, $2);   
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

variableStatement: noop {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| dataType {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| optionalDataType {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| variableInstancePhase {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| instanceMapping {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| connection {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| inAttrPSet {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| outAttrPSet {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| userFunction {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| predicateFunction {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| triggeredFunctionInstance {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

edgeConnection: CONNECTION PRE NODE '(' ')' EXPECTS interfacePointerList '{' edgeConnectionStatementList '}' {
   $$ = new C_edgeConnection($7, $9, Connection::_PRE);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| CONNECTION POST NODE '(' ')' EXPECTS interfacePointerList '{' edgeConnectionStatementList '}' {
   $$ = new C_edgeConnection($7, $9, Connection::_POST);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| CONNECTION PRE edgeConnectionComponentType '(' ')' EXPECTS interfacePointerList '{' connectionStatementList '}' {
   $$ = new C_regularConnection($7, $9,
				$3,
				Connection::_PRE);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| CONNECTION PRE edgeConnectionComponentType '(' predicate ')' EXPECTS interfacePointerList '{' connectionStatementList '}' {
   $$ = new C_regularConnection($8, $10,
				$3,
				Connection::_PRE, $5);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
; 

edgeConnectionComponentType: CONSTANT {
   $$ = Connection::_CONSTANT;
}
| VARIABLE {
   $$ = Connection::_VARIABLE;
}
;


edgeConnectionStatementList: edgeConnectionStatement {
   $$ = new C_generalList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| edgeConnectionStatementList edgeConnectionStatement {
   $$ = new C_generalList($1, $2);   
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

edgeConnectionStatement: noop {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| interfaceToMember {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

connection: CONNECTION PRE connectionComponentType '(' ')' EXPECTS interfacePointerList '{' connectionStatementList '}' {
   $$ = new C_regularConnection($7, $9,
				$3,
				Connection::_PRE);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| CONNECTION PRE connectionComponentType '(' predicate ')' EXPECTS interfacePointerList '{' connectionStatementList '}' {
   $$ = new C_regularConnection($8, $10,
				$3,
				Connection::_PRE, $5);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
; 

connectionComponentType: NODE {
   $$ = Connection::_NODE;
}
| EDGE {
   $$ = Connection::_EDGE;
}
| CONSTANT {
   $$ = Connection::_CONSTANT;
}
| VARIABLE {
   $$ = Connection::_VARIABLE;
}
;

connectionStatementList: connectionStatement {
   $$ = new C_generalList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| connectionStatementList connectionStatement {
   $$ = new C_generalList($1, $2);   
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

connectionStatement: noop {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| interfaceToMember {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| psetToMember {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| userFunctionCall {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

interfaceToMember: IDENTIFIER DOT IDENTIFIER RIGHTSHIFT identifierDotList ';' {
   $$ = new C_interfaceToInstance(*$1, *$3, $5);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
   delete $3;
}
| IDENTIFIER DOT IDENTIFIER RIGHTSHIFT SHARED DOT identifierDotList ';' {
   $$ = new C_interfaceToShared(*$1, *$3, $7);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
   delete $3;
}
;

psetToMember: PSET DOT IDENTIFIER RIGHTSHIFT identifierDotList ';' {
   $$ = new C_psetToInstance(*$3, $5);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $3;
}
| PSET DOT IDENTIFIER RIGHTSHIFT SHARED DOT identifierDotList ';' {
   $$ = new C_psetToShared(*$3, $7);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $3;
}
;

constant: CONSTANT IDENTIFIER IMPLEMENTS interfacePointerList '{' constantStatementList '}' {
   $$ = new C_constant(*$2, $4, $6);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
}
| CONSTANT IDENTIFIER '{' constantStatementList '}' {
   $$ = new C_constant(*$2, 0, $4);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $2;
}
;

constantStatementList: constantStatement {
   $$ = new C_generalList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| constantStatementList constantStatement {
   $$ = new C_generalList($1, $2);   
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

constantStatement: noop {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| dataType {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| instanceMapping {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| outAttrPSet {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

userFunction: USERFUNCTION identifierList ';' {
   $$ = new C_userFunction($2);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

userFunctionCall: IDENTIFIER '(' ')' ';' {
   $$ = new C_userFunctionCall(*$1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
;

predicateFunction: PREDICATEFUNCTION identifierList ';' {
   $$ = new C_predicateFunction($2);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

identifierList: IDENTIFIER {
   $$ = new C_identifierList(*$1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
| identifierList ',' IDENTIFIER {
   $$ = new C_identifierList($1, *$3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $3;
}
;

identifierDotList: IDENTIFIER {
   $$ = new C_identifierList(*$1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
| identifierDotList DOT IDENTIFIER {
   $$ = new C_identifierList($1, *$3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $3;
}
;

phaseIdentifier: IDENTIFIER {
   $$ = new C_phaseIdentifier(*$1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
| IDENTIFIER '(' ')' {
   $$ = new C_phaseIdentifier(*$1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
| IDENTIFIER '(' identifierList ')' {
   $$ = new C_phaseIdentifier(*$1, $3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
;

phaseIdentifierList: phaseIdentifier {
   $$ = new C_phaseIdentifierList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| phaseIdentifierList ',' phaseIdentifier {
   $$ = new C_phaseIdentifierList($1, $3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

edgeInstancePhase: INITPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_initPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| RUNTIMEPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_runtimePhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| FINALPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_finalPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| LOADPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_loadPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

variableInstancePhase: INITPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_initPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| RUNTIMEPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_runtimePhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| FINALPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_finalPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| LOADPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_loadPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

nodeInstancePhase: INITPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_initPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| RUNTIMEPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_runtimePhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| RUNTIMEPHASE GRIDLAYERS phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeGridLayers());
   $$ = new C_runtimePhase($3, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| FINALPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_finalPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| LOADPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   $$ = new C_loadPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

sharedPhase: INITPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   $$ = new C_initPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| RUNTIMEPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   $$ = new C_runtimePhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| FINALPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   $$ = new C_finalPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| LOADPHASE phaseIdentifierList ';' {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   $$ = new C_loadPhase($2, std::move(pType));
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

interfacePointer: IDENTIFIER {
   $$ = new C_interfacePointer(*$1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
;

interfacePointerList: interfacePointer {
   $$ = new C_interfacePointerList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| interfacePointerList ',' interfacePointer {
   $$ = new C_interfacePointerList($1,$3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

instanceMapping: IDENTIFIER DOT IDENTIFIER LEFTSHIFT identifierDotList ';' {
   $$ = new C_instanceMapping(*$1, *$3, $5);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
   delete $3;
}
| IDENTIFIER DOT IDENTIFIER LEFTSHIFT AMPERSAND identifierDotList ';' {
   $$ = new C_instanceMapping(*$1, *$3, $6, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
   delete $3;
} 
;

sharedMapping: IDENTIFIER DOT IDENTIFIER LEFTSHIFT SHARED DOT identifierDotList ';' {
   $$ = new C_sharedMapping(*$1, *$3, $7);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
   delete $3;
}
| IDENTIFIER DOT IDENTIFIER LEFTSHIFT AMPERSAND SHARED DOT identifierDotList ';' {
   $$ = new C_sharedMapping(*$1, *$3, $8, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
   delete $3;
}

;

predicate: '(' predicate ')' {
   $$ = new Predicate(new ParanthesisOp(), $2);  
}
| IDENTIFIER {
   $$ = new InstancePredicate(new TerminalOp(), *$1);  
   delete $1;
}
| IDENTIFIER '(' ')' {
   $$ = new FunctionPredicate(new TerminalOp(), *$1);  
   delete $1;
}
| SHARED DOT IDENTIFIER {
   $$ = new SharedPredicate(new TerminalOp(), *$3);  
   delete $3;
}
| PSET DOT IDENTIFIER {
   $$ = new PSetPredicate(new TerminalOp(), *$3);  
   delete $3;
}
| STRING_LITERAL {
   std::ostringstream os;
   os << '"' << *$1 << '"';
   $$ = new Predicate(new TerminalOp(), os.str(), "string");  
   delete $1;
}
| INT_CONSTANT {
   std::ostringstream os;
   os << $1;
   $$ = new Predicate(new TerminalOp(), os.str(), "int");  
}
| DOUBLE_CONSTANT {
   std::ostringstream os;
   os << $1;
   $$ = new Predicate(new TerminalOp(), os.str(), "double");  
}
| _TRUE {
   $$ = new Predicate(new TerminalOp(), "true", "bool");  
}
| _FALSE {
   $$ = new Predicate(new TerminalOp(), "false", "bool");  
}
| predicate EQUAL predicate {
   $$ = new Predicate(new EqualOp(), $1, $3);  
}
| predicate NOT_EQUAL predicate {
   $$ = new Predicate(new NotEqualOp(), $1, $3);  
}
| predicate LESS_EQUAL predicate {
   $$ = new Predicate(new LessEqualOp(), $1, $3);  
}
| predicate GREATER_EQUAL predicate {
   $$ = new Predicate(new GreaterEqualOp(), $1, $3);  
}
| predicate LESS predicate {
   $$ = new Predicate(new LessOp(), $1, $3);  
}
| predicate GREATER predicate {
   $$ = new Predicate(new GreaterOp(), $1, $3);  
}
| predicate AND predicate {
   $$ = new Predicate(new AndOp(), $1, $3);  
}
| predicate OR predicate {
   $$ = new Predicate(new OrOp(), $1, $3);  
}
;

dataTypeList: dataType {
   $$ = new C_dataTypeList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| dataTypeList dataType {
   $$ = new C_dataTypeList($1, $2);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

optionalDataType: OPTIONAL nonPointerTypeClassifier nameCommentList ';' {
   $$ = new C_dataType($2, $3, false, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| OPTIONAL DERIVED nonPointerTypeClassifier nameCommentList ';' {
   $$ = new C_dataType($3, $4, true, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

dataType: typeClassifier nameCommentList ';' {
   $$ = new C_dataType($1, $2);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| DERIVED typeClassifier nameCommentList ';' {
   $$ = new C_dataType($2, $3, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

nameComment: IDENTIFIER {
   $$ = new C_nameComment(*$1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
| IDENTIFIER '(' INT_CONSTANT ')' {
   $$ = new C_nameComment(*$1, $3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
| IDENTIFIER '(' INT_CONSTANT ',' INT_CONSTANT ')' {
   $$ = new C_nameComment(*$1, $3, $5);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
| IDENTIFIER ':' STRING_LITERAL {
   $$ = new C_nameComment(*$1, *$3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
   delete $3;
}
| IDENTIFIER '(' INT_CONSTANT ')' ':' STRING_LITERAL {
   $$ = new C_nameComment(*$1, *$6, $3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
   delete $6;
}
| IDENTIFIER '(' INT_CONSTANT ',' INT_CONSTANT ')'  ':' STRING_LITERAL {
   $$ = new C_nameComment(*$1, *$8, $3, $5);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
   delete $8;
}
;

nameCommentList: nameComment {   
   $$ = new C_nameCommentList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| nameCommentList ',' nameComment {
   $$ = new C_nameCommentList($1, $3);   
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

nameCommentArgument: IDENTIFIER {
   $$ = new C_nameComment(*$1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
;

nameCommentArgumentList: nameCommentArgument {
   $$ = new C_nameCommentList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

functor: FUNCTOR IDENTIFIER '{' functorStatementList '}' {
    $$ = new C_functor(*$2, $4);
    $$->setTokenLocation(CURRENTFILE, @1.first_line);
    delete $2;
}
| FRAMEWORK FUNCTOR IDENTIFIER '{' functorStatementList '}' {
    $$ = new C_functor(*$3, $5);
    $$->setTokenLocation(CURRENTFILE, @1.first_line);
    $$->setFrameWorkElement();
    delete $3;
}
| FUNCTOR IDENTIFIER CATEGORY STRING_LITERAL '{' functorStatementList '}' {
    $$ = new C_functor(*$2, $6, *$4);
    $$->setTokenLocation(CURRENTFILE, @1.first_line);
    delete $2;
    delete $4;
}
| FRAMEWORK FUNCTOR IDENTIFIER CATEGORY STRING_LITERAL '{' functorStatementList '}' {
    $$ = new C_functor(*$3, $7, *$5);
    $$->setTokenLocation(CURRENTFILE, @1.first_line);
    $$->setFrameWorkElement();
    delete $3;
    delete $5;
}
;

functorStatementList: functorStatement {
   $$ = new C_generalList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| functorStatementList functorStatement {
   $$ = new C_generalList($1, $2);   
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

functorStatement: noop {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| execute {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| initialize {
   $$ = $1;
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

execute: returnType EXECUTE '(' ')' ';' {
   $$ = new C_execute($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);  
} 
| returnType EXECUTE '(' ELLIPSIS ')' ';' {
   $$ = new C_execute($1, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);  
} 
| returnType EXECUTE '(' argumentDataTypeList ')' ';' {
   $$ = new C_execute($1, $4);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);  
} 
| returnType EXECUTE '(' argumentDataTypeList ',' ELLIPSIS ')' ';' {
   $$ = new C_execute($1, $4, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);  
} 
;

returnType: VOID {
   $$ = new C_returnType(true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);    
}
| typeClassifier {
   $$ = new C_returnType($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);    
}
;

initialize: INITIALIZE '(' ')' ';' {
   $$ = new C_initialize();
   $$->setTokenLocation(CURRENTFILE, @1.first_line);  
} 
| INITIALIZE '(' ELLIPSIS ')' ';' {
   $$ = new C_initialize(true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);  
} 
| INITIALIZE '(' argumentDataTypeList ')' ';' {
   $$ = new C_initialize($3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);  
} 
| INITIALIZE '(' argumentDataTypeList ',' ELLIPSIS ')' ';' {
   $$ = new C_initialize($3, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);  
} 
;

argumentDataTypeList: argumentDataType {
   $$ = new C_generalList($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| argumentDataTypeList ',' argumentDataType {
   $$ = new C_generalList($1, $3);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

argumentDataType: typeClassifier nameCommentArgumentList {
   $$ = new C_dataType($1, $2);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

triggeredFunctionInstance: TRIGGEREDFUNCTION identifierList ';' {
   $$ = new C_triggeredFunctionInstance($2, TriggeredFunction::_SERIAL);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| SERIAL TRIGGEREDFUNCTION identifierList ';' {
   $$ = new C_triggeredFunctionInstance($3, TriggeredFunction::_SERIAL);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| PARALLEL TRIGGEREDFUNCTION identifierList ';' {
   $$ = new C_triggeredFunctionInstance($3, TriggeredFunction::_PARALLEL);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

triggeredFunctionShared: TRIGGEREDFUNCTION identifierList ';' {
   $$ = new C_triggeredFunctionShared($2, TriggeredFunction::_SERIAL);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| SERIAL TRIGGEREDFUNCTION identifierList ';' {
   $$ = new C_triggeredFunctionShared($3, TriggeredFunction::_SERIAL);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| PARALLEL TRIGGEREDFUNCTION identifierList ';' {
   $$ = new C_triggeredFunctionShared($3, TriggeredFunction::_PARALLEL);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

// This is tricky, array can have a typeClassifier internally, because
// array of pointers are not prohibited. This results in awkward error 
// reporting when the user does something like "Optional int* eta"
// Here eta is shown as offending instead of *, because * is valid, the
// user could have wrote "int*[] eta" as a legitimite statement.
nonPointerTypeClassifier: typeCore {
   $$ = new C_typeClassifier($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| array { 
   $$ = new C_typeClassifier($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

typeClassifier: typeCore {
   $$ = new C_typeClassifier($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| typeCore STAR {
   $$ = new C_typeClassifier($1, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| array { 
   $$ = new C_typeClassifier($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| array STAR {
   $$ = new C_typeClassifier($1, true);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

typeCore: STRING {
   $$ = new C_typeCore(new StringType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| BOOL {
   $$ = new C_typeCore(new BoolType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| CHAR {
   $$ = new C_typeCore(new CharType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| SHORT {
   $$ = new C_typeCore(new ShortType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| INT {
   $$ = new C_typeCore(new IntType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| LONG {
   $$ = new C_typeCore(new LongType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| FLOAT {
   $$ = new C_typeCore(new FloatType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| DOUBLE {
   $$ = new C_typeCore(new DoubleType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| LONG DOUBLE {
   $$ = new C_typeCore(new LongDoubleType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| UNSIGNED {
   $$ = new C_typeCore(new UnsignedType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| EDGE {
   $$ = new C_typeCore(new EdgeType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| EDGESET {
   $$ = new C_typeCore(new EdgeSetType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| EDGETYPE {
   $$ = new C_typeCore(new EdgeTypeType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| FUNCTOR {
   $$ = new C_typeCore(new FunctorType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| GRID {
   $$ = new C_typeCore(new GridType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| NODE {
   $$ = new C_typeCore(new NodeType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| NODESET {
   $$ = new C_typeCore(new NodeSetType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| NODETYPE {
   $$ = new C_typeCore(new NodeTypeType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| SERVICE {
   $$ = new C_typeCore(new ServiceType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| REPERTOIRE {
   $$ = new C_typeCore(new RepertoireType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| TRIGGER {
   $$ = new C_typeCore(new TriggerType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| PARAMETERSET {
   $$ = new C_typeCore(new ParameterSetType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| NDPAIRLIST {
   $$ = new C_typeCore(new NDPairListType());
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
| IDENTIFIER {
   $$ = new C_typeCore(*$1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
   delete $1;
}
;

array : typeClassifier '[' ']' {
   $$ = new C_array($1);
   $$->setTokenLocation(CURRENTFILE, @1.first_line);
}
;

%%
 
inline int mdllex(YYSTYPE *lvalp, YYLTYPE *locp, void *context)
{
   return ((MdlContext *) context)->_lexer->lex(lvalp, locp, (MdlContext *) context);
}

int main(int argc, char *argv[])
{
  std::string current_date; 
  std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
  std::time_t time = std::chrono::system_clock::to_time_t(tp);
  std::tm* timetm = std::localtime(&time);
  char date_time_format[] = "%m-%d-%Y";
  char time_str[] = "mm-dd-yyyyaa";
  strftime(time_str, strlen(time_str), date_time_format, timetm);
  char year_format[] = "%Y";
  char year[] = "mm-dd-yyyya";
  strftime(year, strlen(year), year_format, timetm);

std::cout << ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n"
          << ".                                                           .\n"
          << ".  Licensed Materials - Property of IBM                     .\n"
          << ".                                                           .\n"
          << ".  \"Restricted Materials of IBM\"                            .\n"
          << ".                                                           .\n"
          //<< ".  BCM-YKT-07-18-2017                                     .\n"
	  << ".  BCM-YKT-"
	  << time_str << "                                     .\n"
          << ".                                                           .\n"
          //<< ".  (C) Copyright IBM Corp. 2005-2017  All rights reserved   .\n"
	  << ".  (C) Copyright IBM Corp. 2005-"
	  << year << "  All rights reserved   .\n"
          << ".                                                           .\n"
          << ".                                                           .\n"
          << ".                                                           .\n"
          << ".  This product includes software developed by the          .\n"
          << ".  University of California, Berkeley and its contributors  .\n"
          << ".                                                           .\n"
          << ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n\n\n";

   Initializer init(argc, argv);
   if (init.execute()) {
      return 0;
   } else {
      return 1;
   }
}

void mdlerror(const char *s)
{
   fprintf(stderr,"%s\n",s);
}

void mdlerror(YYLTYPE*, void*, const char *s)
{
   fprintf(stderr, "%s\n", s);
}
