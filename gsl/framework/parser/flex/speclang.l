%{
// =============================================================================*/
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)                        */
//                                                                  */
// =============================================================================*/
%}
%x COMMENT

%option c++
%option yyclass="GslLexer"
%option noyywrap

/* Digits, letters and alphanums */
D       [0-9]
L       [a-zA-Z]
A       [a-zA-Z0-9]
U       [a-zA-Z0-9_]

%{
//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>
//#include <sstream>
#include <fstream>
#include <string>
#include "GslLexer.h"
using namespace std;

%}

%%
"//".*            { /* Ignore C++-style comments */ }
"/*"              { BEGIN(COMMENT); }
<COMMENT>"*/"     { BEGIN(INITIAL); }
<COMMENT>.|\n     { /* Ignore everything in a comment */ }
int               { yylloc->first_line = lineCount;
                    return ( LOCAL_INT );
                    }
string            { yylloc->first_line = lineCount;
                    return ( STRING );
                    }
float             { yylloc->first_line = lineCount;
                    return ( LOCAL_FLOAT );
                    }
list              { yylloc->first_line = lineCount;
                    return ( LIST );
                    }
Type              { yylloc->first_line = lineCount;
                    return ( TYPE );
                    }
RefPointGen       { yylloc->first_line = lineCount;
                    return ( REFPTGEN );
                    }
SamplingFctr1     { yylloc->first_line = lineCount;
                    return ( SAMPFCTR1 );
                    }
SamplingFctr2     { yylloc->first_line = lineCount;
                    return ( SAMPFCTR2 );
                    }
SubNodeSetFctr    { yylloc->first_line = lineCount;
                    return ( SUBNODESETFCTR );
                    }
Layout            { yylloc->first_line = lineCount;
                    return ( LAYOUT );
                    }
NodeInitializer   { yylloc->first_line = lineCount;
                    return ( NODEINITIALIZER );
                    }
InitNodes         { yylloc->first_line = lineCount;
                    return ( INITNODES );
                    }
EdgeInitializer   { yylloc->first_line = lineCount;
                    return ( EDGEINITIALIZER );
                    }
InAttrInitializer   { yylloc->first_line = lineCount;
                    return ( INATTRINITIALIZER );
                    }
NDPairListFunctor { yylloc->first_line = lineCount;
                    return ( NDPAIRLISTFUNCTOR);
                    }
NDPairList        { yylloc->first_line = lineCount;
                    return ( NDPAIRLIST);
                    }
Complex           { yylloc->first_line = lineCount;
                    return ( LOCAL_COMPLEX );
                    }
"||"              { yylloc->first_line = lineCount;
                    return ( EXP_OR );
                    }
"^^"              { yylloc->first_line = lineCount;
                    return ( EXP_XOR );
                    }
"&&"              { yylloc->first_line = lineCount;
                    return ( EXP_AND );
                    }
"=="              { yylloc->first_line = lineCount;
                    return ( EQUIVALENT );
                    }
"!="              { yylloc->first_line = lineCount;
                    return ( NOT_EQUIVALENT );
                    }
Member            { yylloc->first_line = lineCount;
                    return ( MEMBER );
                    }
TypeDef           { yylloc->first_line = lineCount;
                    return ( TYPEDEF );
                    }
Initialize        { yylloc->first_line = lineCount;
                    return ( INITIALIZE );
                    }
Function          { yylloc->first_line = lineCount;
                    return ( FUNCTION );
                    }
Functor           { yylloc->first_line = lineCount;
                    return ( FUNCTOR );
                    }
Dimension         { yylloc->first_line = lineCount;
                    return ( DIMENSION );
                    }
NodeIndex         { yylloc->first_line = lineCount;
                    return ( NODEINDEX );
                    }
EdgeIndex         { yylloc->first_line = lineCount;
                    return ( EDGEINDEX );
                    }
Default           { yylloc->first_line = lineCount;
                    return ( DEFAULT );
                    }
NodeInit          { yylloc->first_line = lineCount;
                    return ( NODEINIT );
                    }
Out               { yylloc->first_line = lineCount;
                    return ( OUT );
                    }
EdgeInit          { yylloc->first_line = lineCount;
                    return ( EDGEINIT );
                    }
In                { yylloc->first_line = lineCount;
                    return ( IN );
                    }
Matrix            { yylloc->first_line = lineCount;
                    return ( MATRIX );
                    }
NDPair            { yylloc->first_line = lineCount;
                    return ( NDPAIR );
                    }
Pset              { yylloc->first_line = lineCount;
                    return ( PSET );
                    }
Grid              { yylloc->first_line = lineCount;
                    return ( GRID );
                    }
Composite         { yylloc->first_line = lineCount;
                    return ( COMPOSITE );
                    }
Connector         { yylloc->first_line = lineCount;
                    return ( CONNECTOR );
                    }
ConnectionScript  { yylloc->first_line = lineCount;
                    return ( CONNECTIONSCRIPT );
                    }
GridCoord         { yylloc->first_line = lineCount;
                    return ( GRIDCOORD );
                    }
RepName           { yylloc->first_line = lineCount;
                    return ( REPNAME );
                    }
NodeSet           { yylloc->first_line = lineCount;
                    return ( NODESET );
                    }
ConstantType      { yylloc->first_line = lineCount;
                    return ( CONSTANTTYPE );
                    }
VariableType      { yylloc->first_line = lineCount;
                    return ( VARIABLETYPE );
                    }
NodeType          { yylloc->first_line = lineCount;
                    return ( NODETYPE );
                    }
NodeTypeSet       { yylloc->first_line = lineCount;
                    return ( NODETYPESET );
                    }
EdgeType          { yylloc->first_line = lineCount;
                    return ( EDGETYPE );
                    }
ToolType          { yylloc->first_line = lineCount;
                    return ( TOOLTYPE );
                    }
Struct            { yylloc->first_line = lineCount;
                    return ( STRUCT );
                    }
"/"               { yylloc->first_line = lineCount;
                    return ( SLASH );
                    }
"."               { yylloc->first_line = lineCount;
                    return ( DOT );
                    }
"..."             { yylloc->first_line = lineCount;
                    return ( ELLIPSIS );
                    }
Stride            { yylloc->first_line = lineCount;
                    return ( STRIDE );
                    }
Layer             { yylloc->first_line = lineCount;
                    return ( LAYER );
                    }
Return            { yylloc->first_line = lineCount;
                    return ( RETURN );
                    }
IndexSet          { yylloc->first_line = lineCount;
                    return ( INDEXSET );
                    }
RelNodeSet        { yylloc->first_line = lineCount;
                    return ( RELNODESET );
                    }
Service           { yylloc->first_line = lineCount;
                    return ( SERVICE );
                    }
Trigger           { yylloc->first_line = lineCount;
                    return ( TRIGGER );
                    }
GranuleMapper     { yylloc->first_line = lineCount;
                    return ( GRANULEMAPPER );
                    }
EdgeSet           { yylloc->first_line = lineCount;
                    return ( EDGESET );
                    }
Port              { yylloc->first_line = lineCount;
                    return ( PORT );
                    }
system            { yylloc->first_line = lineCount;
                    return ( SYSTEM );
                    }
Pause             { yylloc->first_line = lineCount;
                    return ( PAUSE );
                    }
Stop              { yylloc->first_line = lineCount;
                    return ( STOP );
                    }
InitPhases        { yylloc->first_line = lineCount;
                    return ( INITPHASES );
                    }
RuntimePhases     { yylloc->first_line = lineCount;
                    return ( RUNTIMEPHASES );
                    }
LoadPhases        { yylloc->first_line = lineCount;
                    return ( LOADPHASES );
                    }
FinalPhases       { yylloc->first_line = lineCount;
                    return ( FINALPHASES );
                    }
SeperationConstraint { yylloc->first_line = lineCount;
                    return ( SEPARATIONCONSTRAINT );
                    }
on                { yylloc->first_line = lineCount;
                    return ( ON );
                    }
"->"              { yylloc->first_line = lineCount;
                    return ( ARROW );
                    }
":"               { yylloc->first_line = lineCount;
                    return ( COLON );
                    }
"::"              { yylloc->first_line = lineCount;
                    return ( DOUBLE_COLON );
                    }
":::"             { yylloc->first_line = lineCount;
                    return ( TRIPLE_COLON );
                    }
"-"               { yylloc->first_line = lineCount;
                    return ( MINUS );
                    }
"+"		  { yylloc->first_line = lineCount;
                    return ( PLUS );
                    }
\"[^\"\n]+\"      { yylloc->first_line = lineCount;
                    yylval->P_string = new std::string(yytext); 
                    std::string::iterator b;
                    b = yylval->P_string->begin(); 
                    yylval->P_string->erase(b);
                    std::string::iterator e;
                    e = yylval->P_string->end(); 
                    yylval->P_string->erase(--e);
                    //cout << "yytext = " << *yylval->P_string << endl;
                    return ( STRING_LITERAL ); 
                    }
{L}{A}*           { yylloc->first_line = lineCount;
                    yylval->P_string = new std::string (yytext); 
                    //cout << "yytext = " << yytext << endl; 
                    return(IDENTIFIER); 
                    }
{L}[_]{A}*        { yylloc->first_line = lineCount;
                    yylval->P_string = new std::string (yytext); 
                    //cout << "yytext = " << yytext << endl; 
                    return(IDENTIFIER); 
                    }
{L}{A}*[_]*{A}*   { yylloc->first_line = lineCount;
                    yylval->P_string = new std::string (yytext); 
                    //cout << "yytext = " << yytext << endl; 
                    return(IDENTIFIER); 
                    }
{L}{A}*{U}*   { yylloc->first_line = lineCount;
                    yylval->P_string = new std::string (yytext); 
                    //cout << "yytext = " << yytext << endl; 
                    return(IDENTIFIER); 
                    }
{D}{A}*{L}{A}*    { yylloc->first_line = lineCount;
                    yylval->P_string = new std::string (yytext);  
                    //cout << "yytext = " << yytext << endl;
                    return(IDENTIFIER); 
                    }
\-?{D}+           { yylloc->first_line = lineCount;
                    yylval->V_int= atoi (yytext); 
                    //cout << "yytext = " << yytext << endl;
                    return(INT_CONSTANT); 
                    }
[+\-]?{D}+\.?{D}*[eE][+\-]?{D}+  { 
                    yylloc->first_line = lineCount;
                    yylval->V_double= atof (yytext); 
                    //cout << "yytext = " << yytext << endl;
                    return(FLOAT_CONSTANT); 
                    }
[+\-]?{D}+\.{D}*  { yylloc->first_line = lineCount;
                    yylval->V_double= atof (yytext);  
                    //cout << "yytext = " << yytext << endl;
                    return(FLOAT_CONSTANT); 
                    }
"\n"              { lineCount++; 
                    //cerr<< "\n\nFinished " <<currentFileName<<": "<< lineCount<<endl;
                    }
[\t ]+            ;
\#                { skip_proc(); 
                    }
.                 { yylloc->first_line = lineCount;
                    return (yytext[0]); 
                    }
%%

int yywrap(void) {
 return 1;
}
