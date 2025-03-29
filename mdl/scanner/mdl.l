%{
/* =================================================================*/
/* Licensed Materials - Property of IBM                             */
/*                                                                  */
/* "Restricted Materials of IBM"                                    */
/*                                                                  */
/* BCM-YKT-07-18-2017                                             */
/*                                                                  */
/* (C) Copyright IBM Corp. 2005-2017  All rights reserved          */
/*                                                                  */
/* US Government Users Restricted Rights -                          */
/* Use, duplication or disclosure restricted by                     */
/* GSA ADP Schedule Contract with IBM Corp.                         */
/*                                                                  */
/* =================================================================*/
%}
%{
/* License header */
#include "bison_compat.h"  // Include the compatibility definitions
#include "mdl_types.h"
#include "ParserClasses.h"
#include "MdlLexer.h"
#include "MdlContext.h"
#include "mdl.tab.h"       // Include the parser definitions

#include <iostream>
#include <string>
using namespace std;

// Define token macros for lexer

#define TOKEN_MDLEMPTY -2
#define TOKEN_MDLEOF 0                    /* "end of file"  */
#define TOKEN_MDLerror 256                /* error  */
#define TOKEN_MDLUNDEF 257                /* "invalid token"  */
#define TOKEN_DOUBLE_CONSTANT 258         /* DOUBLE_CONSTANT  */
#define TOKEN_INT_CONSTANT 259            /* INT_CONSTANT  */
#define TOKEN_STRING_LITERAL 260          /* STRING_LITERAL  */
#define TOKEN_IDENTIFIER 261              /* IDENTIFIER  */
#define TOKEN_STRING 262                  /* STRING  */
#define TOKEN_BOOL 263                    /* BOOL  */
#define TOKEN_CHAR 264                    /* CHAR  */
#define TOKEN_SHORT 265                   /* SHORT  */
#define TOKEN_INT 266                     /* INT  */
#define TOKEN_LONG 267                    /* LONG  */
#define TOKEN_FLOAT 268                   /* FLOAT  */
#define TOKEN_DOUBLE 269                  /* DOUBLE  */
#define TOKEN_UNSIGNED 270                /* UNSIGNED  */
#define TOKEN_EDGE 271                    /* EDGE  */
#define TOKEN_EDGESET 272                 /* EDGESET  */
#define TOKEN_EDGETYPE 273                /* EDGETYPE  */
#define TOKEN_FUNCTOR 274                 /* FUNCTOR  */
#define TOKEN_GRID 275                    /* GRID  */
#define TOKEN_NODE 276                    /* NODE  */
#define TOKEN_NODESET 277                 /* NODESET  */
#define TOKEN_NODETYPE 278                /* NODETYPE  */
#define TOKEN_SERVICE 279                 /* SERVICE  */
#define TOKEN_REPERTOIRE 280              /* REPERTOIRE  */
#define TOKEN_TRIGGER 281                 /* TRIGGER  */
#define TOKEN_TRIGGEREDFUNCTION 282       /* TRIGGEREDFUNCTION  */
#define TOKEN_SERIAL 283                  /* SERIAL  */
#define TOKEN_PARALLEL 284                /* PARALLEL  */
#define TOKEN_PARAMETERSET 285            /* PARAMETERSET  */
#define TOKEN_NDPAIRLIST 286              /* NDPAIRLIST  */
#define TOKEN_OR 287                      /* OR  */
#define TOKEN_AND 288                     /* AND  */
#define TOKEN_EQUAL 289                   /* EQUAL  */
#define TOKEN_NOT_EQUAL 290               /* NOT_EQUAL  */
#define TOKEN_LESS_EQUAL 291              /* LESS_EQUAL  */
#define TOKEN_GREATER_EQUAL 292           /* GREATER_EQUAL  */
#define TOKEN_LESS 293                    /* LESS  */
#define TOKEN_GREATER 294                 /* GREATER  */
#define TOKEN_DOT 295                     /* DOT  */
#define TOKEN_AMPERSAND 296               /* AMPERSAND  */
#define TOKEN_LEFTSHIFT 297               /* LEFTSHIFT  */
#define TOKEN_RIGHTSHIFT 298              /* RIGHTSHIFT  */
#define TOKEN_ELLIPSIS 299                /* ELLIPSIS  */
#define TOKEN_STAR 300                    /* STAR  */
#define TOKEN__TRUE 301                   /* _TRUE  */
#define TOKEN__FALSE 302                  /* _FALSE  */
#define TOKEN_DERIVED 303                 /* DERIVED  */
#define TOKEN_STRUCT 304                  /* STRUCT  */
#define TOKEN_INTERFACE 305               /* INTERFACE  */
#define TOKEN_CONNECTION 306              /* CONNECTION  */
#define TOKEN_PRENODE 307                 /* PRENODE  */
#define TOKEN_POSTNODE 308                /* POSTNODE  */
#define TOKEN_EXPECTS 309                 /* EXPECTS  */
#define TOKEN_IMPLEMENTS 310              /* IMPLEMENTS  */
#define TOKEN_SHARED 311                  /* SHARED  */
#define TOKEN_INATTRPSET 312              /* INATTRPSET  */
#define TOKEN_OUTATTRPSET 313             /* OUTATTRPSET  */
#define TOKEN_PSET 314                    /* PSET  */
#define TOKEN_INITPHASE 315               /* INITPHASE  */
#define TOKEN_RUNTIMEPHASE 316            /* RUNTIMEPHASE  */
#define TOKEN_FINALPHASE 317              /* FINALPHASE  */
#define TOKEN_LOADPHASE 318               /* LOADPHASE  */
#define TOKEN_CONSTANT 319                /* CONSTANT  */
#define TOKEN_VARIABLE 320                /* VARIABLE  */
#define TOKEN_USERFUNCTION 321            /* USERFUNCTION  */
#define TOKEN_PREDICATEFUNCTION 322       /* PREDICATEFUNCTION  */
#define TOKEN_INITIALIZE 323              /* INITIALIZE  */
#define TOKEN_EXECUTE 324                 /* EXECUTE  */
#define TOKEN_CATEGORY 325                /* CATEGORY  */
#define TOKEN_VOID 326                    /* VOID  */
#define TOKEN_PRE 327                     /* PRE  */
#define TOKEN_POST 328                    /* POST  */
#define TOKEN_GRIDLAYERS 329              /* GRIDLAYERS  */
#define TOKEN_THREADS 330                 /* THREADS  */
#define TOKEN_OPTIONAL 331                /* OPTIONAL  */
#define TOKEN_FRAMEWORK 332               /* FRAMEWORK  */

// Helper method declarations
void MdlLexer::skip_proc(void) {
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

const char* MdlLexer::getToken() {
    return yytext;
}

%}

%option c++
%option yyclass="MdlLexer"
%option noyywrap

/* Digits, letters and alphanums */
D       [0-9]
L       [a-zA-Z]
A       [a-zA-Z0-9]
U       [a-zA-Z0-9_]

%%
string { 
   yylloc->first_line = lineCount;
   return (TOKEN_STRING);  
}
bool { 
   yylloc->first_line = lineCount;
   return (TOKEN_BOOL);  
}
char { 
   yylloc->first_line = lineCount;
   return (TOKEN_CHAR);
}
short { 
   yylloc->first_line = lineCount;
   return (TOKEN_SHORT);
}
int { 
   yylloc->first_line = lineCount;
   return (TOKEN_INT);
}
long { 
   yylloc->first_line = lineCount;
   return (TOKEN_LONG);
}
float { 
   yylloc->first_line = lineCount;
   return (TOKEN_FLOAT);
}
double { 
   yylloc->first_line = lineCount;
   return (TOKEN_DOUBLE);
}
unsigned { 
   yylloc->first_line = lineCount;
   return (TOKEN_UNSIGNED);
}
Edge { 
   yylloc->first_line = lineCount;
   return (TOKEN_EDGE);
}
EdgeSet { 
   yylloc->first_line = lineCount;
   return (TOKEN_EDGESET);
}
EdgeType { 
   yylloc->first_line = lineCount;
   return (TOKEN_EDGETYPE);
}
Functor { 
   yylloc->first_line = lineCount;
   return (TOKEN_FUNCTOR);
}
Grid { 
   yylloc->first_line = lineCount;
   return (TOKEN_GRID);
}
Node { 
   yylloc->first_line = lineCount;
   return (TOKEN_NODE);
}
NodeSet { 
   yylloc->first_line = lineCount;
   return (TOKEN_NODESET);
}
NodeType { 
   yylloc->first_line = lineCount;
   return (TOKEN_NODETYPE);
}
Service { 
   yylloc->first_line = lineCount;
   return (TOKEN_SERVICE);
}
Repertoire { 
   yylloc->first_line = lineCount;
   return (TOKEN_REPERTOIRE);
}
Trigger { 
   yylloc->first_line = lineCount;
   return (TOKEN_TRIGGER);
}
TriggeredFunction { 
   yylloc->first_line = lineCount;
   return (TOKEN_TRIGGEREDFUNCTION);
}
Serial { 
   yylloc->first_line = lineCount;
   return (TOKEN_SERIAL);
}
Parallel { 
   yylloc->first_line = lineCount;
   return (TOKEN_PARALLEL);
}
ParameterSet { 
   yylloc->first_line = lineCount;
   return (TOKEN_PARAMETERSET);
}
Optional { 
   yylloc->first_line = lineCount;
   return (TOKEN_OPTIONAL);
}
NDPairList { 
   yylloc->first_line = lineCount;
   return (TOKEN_NDPAIRLIST);
}
Pre { 
   yylloc->first_line = lineCount;
   return (TOKEN_PRE);
}
Post { 
   yylloc->first_line = lineCount;
   return (TOKEN_POST);
}
Framework { 
   yylloc->first_line = lineCount;
   return (TOKEN_FRAMEWORK);
}
"||" { 
   yylloc->first_line = lineCount;
   return (TOKEN_OR);
}
"&&" { 
   yylloc->first_line = lineCount;
   return (TOKEN_AND);
}
"==" { 
   yylloc->first_line = lineCount;
   return (TOKEN_EQUAL);
}
"!=" { 
   yylloc->first_line = lineCount;
   return (TOKEN_NOT_EQUAL);
}
"<=" { 
   yylloc->first_line = lineCount;
   return (TOKEN_LESS_EQUAL);
}
">=" { 
   yylloc->first_line = lineCount;
   return (TOKEN_GREATER_EQUAL);
}
"<" { 
   yylloc->first_line = lineCount;
   return (TOKEN_LESS);
}
">" { 
   yylloc->first_line = lineCount;
   return (TOKEN_GREATER);
}
"." { 
   yylloc->first_line = lineCount;
   return (TOKEN_DOT);
}
"&" { 
   yylloc->first_line = lineCount;
   return (TOKEN_AMPERSAND);
}
"<<" { 
   yylloc->first_line = lineCount;
   return (TOKEN_LEFTSHIFT);
}
">>" { 
   yylloc->first_line = lineCount;
   return (TOKEN_RIGHTSHIFT);
}
"*" { 
   yylloc->first_line = lineCount;
   return (TOKEN_STAR);
}
"..." { 
   yylloc->first_line = lineCount;
   return (TOKEN_ELLIPSIS);
}
true {
   yylloc->first_line = lineCount;
   return (TOKEN__TRUE);
}
false {
   yylloc->first_line = lineCount;
   return (TOKEN__FALSE);
}
Derived {
   yylloc->first_line = lineCount;
   return (TOKEN_DERIVED);
}
Struct {
   yylloc->first_line = lineCount;
   return (TOKEN_STRUCT);
}
Interface {
   yylloc->first_line = lineCount;
   return (TOKEN_INTERFACE);
}
Connection {
   yylloc->first_line = lineCount;
   return (TOKEN_CONNECTION);
}
PreNode {
   yylloc->first_line = lineCount;
   return (TOKEN_PRENODE);
}
PostNode {
   yylloc->first_line = lineCount;
   return (TOKEN_POSTNODE);
}
Expects {
   yylloc->first_line = lineCount;
   return (TOKEN_EXPECTS);
}
Implements {
   yylloc->first_line = lineCount;
   return (TOKEN_IMPLEMENTS);
}
Shared {
   yylloc->first_line = lineCount;
   return (TOKEN_SHARED);
}
InAttrPSet {
   yylloc->first_line = lineCount;
   return (TOKEN_INATTRPSET);
}
OutAttrPSet {
   yylloc->first_line = lineCount;
   return (TOKEN_OUTATTRPSET);
}
PSet {
   yylloc->first_line = lineCount;
   return (TOKEN_PSET);
}
InitPhase {
   yylloc->first_line = lineCount;
   return (TOKEN_INITPHASE);
}
RuntimePhase {
   yylloc->first_line = lineCount;
   return (TOKEN_RUNTIMEPHASE);
}
FinalPhase {
   yylloc->first_line = lineCount;
   return (TOKEN_FINALPHASE);
}
LoadPhase {
   yylloc->first_line = lineCount;
   return (TOKEN_LOADPHASE);
}
Constant {
   yylloc->first_line = lineCount;
   return (TOKEN_CONSTANT);
}
Variable {
   yylloc->first_line = lineCount;
   return (TOKEN_VARIABLE);
}
UserFunction {
   yylloc->first_line = lineCount;
   return (TOKEN_USERFUNCTION);
}
PredicateFunction {
   yylloc->first_line = lineCount;
   return (TOKEN_PREDICATEFUNCTION);
}
Initialize {
   yylloc->first_line = lineCount;
   return (TOKEN_INITIALIZE);
}
Execute {
   yylloc->first_line = lineCount;
   return (TOKEN_EXECUTE);
}
Category {
   yylloc->first_line = lineCount;
   return (TOKEN_CATEGORY);
}
void {
   yylloc->first_line = lineCount;
   return (TOKEN_VOID);
}
GridLayers {
   yylloc->first_line = lineCount;
   return (TOKEN_GRIDLAYERS);
}
Threads {
   yylloc->first_line = lineCount;
   return (TOKEN_THREADS);
}
\"[^\"\n]+\" { 
   yylloc->first_line = lineCount;
   yylval->P_string = new string(yytext); 
   string::iterator b;
   b = yylval->P_string->begin(); 
   yylval->P_string->erase(b);
   string::iterator e;
   e = yylval->P_string->end(); 
   yylval->P_string->erase(--e);
   //cout << "yytext = " << *yylval->P_string << endl;
   return (TOKEN_STRING_LITERAL); 
}
{L}{A}* { 
   yylloc->first_line = lineCount;
   yylval->P_string = new string (yytext); 
   //cout << "yytext = " << yytext << endl; 
   return(IDENTIFIER); 
}
{L}[_]{A}* { 
   yylloc->first_line = lineCount;
   yylval->P_string = new string (yytext); 
   //cout << "yytext = " << yytext << endl; 
   return(IDENTIFIER); 
}
{L}{A}*[_]*{A}* { 
   yylloc->first_line = lineCount;
   yylval->P_string = new string (yytext); 
   //cout << "yytext = " << yytext << endl; 
   return(IDENTIFIER); 
}
{L}{A}*{U}*   { 
   yylloc->first_line = lineCount;
   yylval->P_string = new string (yytext); 
   //cout << "yytext = " << yytext << endl; 
   return(IDENTIFIER); 
}
[_]{L}{A}*{U}*   { 
   yylloc->first_line = lineCount;
   yylval->P_string = new string (yytext); 
   //cout << "yytext = " << yytext << endl; 
   return(IDENTIFIER); 
}
[_]{A}*{U}*   { 
   yylloc->first_line = lineCount;
   yylval->P_string = new string (yytext); 
   //cout << "yytext = " << yytext << endl; 
   return(IDENTIFIER); 
}
\-?{D}+ { 
   yylloc->first_line = lineCount;
   yylval->V_int= atoi (yytext); 
   //cout << "yytext = " << yytext << endl;
   return(INT_CONSTANT); 
}
[+\-]?{D}+\.?{D}*[eE][+\-]?{D}+ { 
   yylloc->first_line = lineCount;
   yylval->V_double= atof (yytext); 
   //cout << "yytext = " << yytext << endl;
   return(DOUBLE_CONSTANT); 
}
[+\-]?{D}+\.{D}* { 
   yylloc->first_line = lineCount;
   yylval->V_double= atof (yytext);  
   //cout << "yytext = " << yytext << endl;
   return(DOUBLE_CONSTANT); 
}

"\n" { 
   lineCount++; 
}
[\t ]+            ;
\# { 
   skip_proc(); 
}
"//"[^\n]*  { 
   /* Skip C++ style comments */
}
"/*"([^*]|"*"[^/])*"*/"  {
   /* Count newlines in block comments */
   for(int i = 0; i < yyleng; i++) {
      if(yytext[i] == '\n') lineCount++;
   }
}
. { 
   yylloc->first_line = lineCount;
   return (yytext[0]); 
}
%%

// Implement constructor/destructor outside of the flex-generated code
MdlLexer::MdlLexer(std::istream* infile, std::ostream* outfile) 
   : yyFlexLexer(infile, outfile) {
   lineCount = 1;
}

MdlLexer::~MdlLexer() {
   // No specific cleanup needed
}

