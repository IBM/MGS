%{
/* =================================================================*/
/* Licensed Materials - Property of IBM                             */
/*                                                                  */
/* "Restricted Materials of IBM"                                    */
/*                                                                  */
/* BMC-YKT-03-01-2006-2                                             */
/*                                                                  */
/* (C) Copyright IBM Corp. 2005-2014  All rights reserved          */
/*                                                                  */
/* US Government Users Restricted Rights -                          */
/* Use, duplication or disclosure restricted by                     */
/* GSA ADP Schedule Contract with IBM Corp.                         */
/*                                                                  */
/* =================================================================*/
%}

%option c++
%option yyclass="MdlLexer"
%option noyywrap

/* Digits, letters and alphanums */
D       [0-9]
L       [a-zA-Z]
A       [a-zA-Z0-9]
U       [a-zA-Z0-9_]

%{

#include <iostream>
#include <string>
#include "MdlLexer.h"
using namespace std;
%}

%%
string { 
   yylloc->first_line = lineCount;
   return ( STRING );  
}
bool { 
   yylloc->first_line = lineCount;
   return ( BOOL );  
}
char { 
   yylloc->first_line = lineCount;
   return ( CHAR );
}
short { 
   yylloc->first_line = lineCount;
   return ( SHORT );
}
int { 
   yylloc->first_line = lineCount;
   return ( INT );
}
long { 
   yylloc->first_line = lineCount;
   return ( LONG );
}
float { 
   yylloc->first_line = lineCount;
   return ( FLOAT );
}
double { 
   yylloc->first_line = lineCount;
   return ( DOUBLE );
}
unsigned { 
   yylloc->first_line = lineCount;
   return ( UNSIGNED );
}
Edge { 
   yylloc->first_line = lineCount;
   return ( EDGE );
}
EdgeSet { 
   yylloc->first_line = lineCount;
   return ( EDGESET );
}
EdgeType { 
   yylloc->first_line = lineCount;
   return ( EDGETYPE );
}
Functor { 
   yylloc->first_line = lineCount;
   return ( FUNCTOR );
}
Grid { 
   yylloc->first_line = lineCount;
   return ( GRID );
}
Node { 
   yylloc->first_line = lineCount;
   return ( NODE );
}
NodeSet { 
   yylloc->first_line = lineCount;
   return ( NODESET );
}
NodeType { 
   yylloc->first_line = lineCount;
   return ( NODETYPE );
}
Service { 
   yylloc->first_line = lineCount;
   return ( SERVICE );
}
Repertoire { 
   yylloc->first_line = lineCount;
   return ( REPERTOIRE );
}
Trigger { 
   yylloc->first_line = lineCount;
   return ( TRIGGER );
}
TriggeredFunction { 
   yylloc->first_line = lineCount;
   return ( TRIGGEREDFUNCTION );
}
Serial { 
   yylloc->first_line = lineCount;
   return ( SERIAL );
}
Parallel { 
   yylloc->first_line = lineCount;
   return ( PARALLEL );
}
ParameterSet { 
   yylloc->first_line = lineCount;
   return ( PARAMETERSET );
}
Optional { 
   yylloc->first_line = lineCount;
   return ( OPTIONAL );
}
NDPairList { 
   yylloc->first_line = lineCount;
   return ( NDPAIRLIST );
}
Pre { 
   yylloc->first_line = lineCount;
   return ( PRE );
}
Post { 
   yylloc->first_line = lineCount;
   return ( POST );
}
Framework { 
   yylloc->first_line = lineCount;
   return ( FRAMEWORK );
}
"||" { 
   yylloc->first_line = lineCount;
   return ( OR );
}
"&&" { 
   yylloc->first_line = lineCount;
   return ( AND );
}
"==" { 
   yylloc->first_line = lineCount;
   return ( EQUAL );
}
"!=" { 
   yylloc->first_line = lineCount;
   return ( NOT_EQUAL );
}
"<=" { 
   yylloc->first_line = lineCount;
   return ( LESS_EQUAL );
}
">=" { 
   yylloc->first_line = lineCount;
   return ( GREATER_EQUAL );
}
"<" { 
   yylloc->first_line = lineCount;
   return ( LESS );
}
">" { 
   yylloc->first_line = lineCount;
   return ( GREATER );
}
"." { 
   yylloc->first_line = lineCount;
   return ( DOT );
}
"&" { 
   yylloc->first_line = lineCount;
   return ( AMPERSAND );
}
"<<" { 
   yylloc->first_line = lineCount;
   return ( LEFTSHIFT );
}
">>" { 
   yylloc->first_line = lineCount;
   return ( RIGHTSHIFT );
}
"*" { 
   yylloc->first_line = lineCount;
   return ( STAR );
}
"..." { 
   yylloc->first_line = lineCount;
   return ( ELLIPSIS );
}
true {
   yylloc->first_line = lineCount;
   return ( _TRUE );
}
false {
   yylloc->first_line = lineCount;
   return ( _FALSE );
}
Derived {
   yylloc->first_line = lineCount;
   return ( DERIVED );
}
Struct {
   yylloc->first_line = lineCount;
   return ( STRUCT );
}
Interface {
   yylloc->first_line = lineCount;
   return ( INTERFACE );
}
Connection {
   yylloc->first_line = lineCount;
   return ( CONNECTION );
}
PreNode {
   yylloc->first_line = lineCount;
   return ( PRENODE );
}
PostNode {
   yylloc->first_line = lineCount;
   return ( POSTNODE );
}
Expects {
   yylloc->first_line = lineCount;
   return ( EXPECTS );
}
Implements {
   yylloc->first_line = lineCount;
   return ( IMPLEMENTS );
}
Shared {
   yylloc->first_line = lineCount;
   return ( SHARED );
}
InAttrPSet {
   yylloc->first_line = lineCount;
   return ( INATTRPSET );
}
OutAttrPSet {
   yylloc->first_line = lineCount;
   return ( OUTATTRPSET );
}
PSet {
   yylloc->first_line = lineCount;
   return ( PSET );
}
InitPhase {
   yylloc->first_line = lineCount;
   return ( INITPHASE );
}
RuntimePhase {
   yylloc->first_line = lineCount;
   return ( RUNTIMEPHASE );
}
FinalPhase {
   yylloc->first_line = lineCount;
   return ( FINALPHASE );
}
LoadPhase {
   yylloc->first_line = lineCount;
   return ( LOADPHASE );
}
Constant {
   yylloc->first_line = lineCount;
   return ( CONSTANT );
}
Variable {
   yylloc->first_line = lineCount;
   return ( VARIABLE );
}
UserFunction {
   yylloc->first_line = lineCount;
   return ( USERFUNCTION );
}
PredicateFunction {
   yylloc->first_line = lineCount;
   return ( PREDICATEFUNCTION );
}
Initialize {
   yylloc->first_line = lineCount;
   return ( INITIALIZE );
}
Execute {
   yylloc->first_line = lineCount;
   return ( EXECUTE );
}
Category {
   yylloc->first_line = lineCount;
   return ( CATEGORY );
}
void {
   yylloc->first_line = lineCount;
   return ( VOID );
}
GridLayers {
   yylloc->first_line = lineCount;
   return ( GRIDLAYERS );
}
Threads {
   yylloc->first_line = lineCount;
   return ( THREADS );
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
   return ( STRING_LITERAL ); 
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
. { 
   yylloc->first_line = lineCount;
   return (yytext[0]); 
}
%%

