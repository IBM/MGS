/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Using locations.  */
#define YYLSP_NEEDED 1



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     DOUBLE_CONSTANT = 258,
     INT_CONSTANT = 259,
     STRING_LITERAL = 260,
     IDENTIFIER = 261,
     STRING = 262,
     BOOL = 263,
     CHAR = 264,
     SHORT = 265,
     INT = 266,
     LONG = 267,
     FLOAT = 268,
     DOUBLE = 269,
     UNSIGNED = 270,
     EDGE = 271,
     EDGESET = 272,
     EDGETYPE = 273,
     FUNCTOR = 274,
     GRID = 275,
     NODE = 276,
     NODESET = 277,
     NODETYPE = 278,
     SERVICE = 279,
     REPERTOIRE = 280,
     TRIGGER = 281,
     TRIGGEREDFUNCTION = 282,
     SERIAL = 283,
     PARALLEL = 284,
     PARAMETERSET = 285,
     NDPAIRLIST = 286,
     OR = 287,
     AND = 288,
     EQUAL = 289,
     NOT_EQUAL = 290,
     LESS_EQUAL = 291,
     GREATER_EQUAL = 292,
     LESS = 293,
     GREATER = 294,
     DOT = 295,
     AMPERSAND = 296,
     LEFTSHIFT = 297,
     RIGHTSHIFT = 298,
     ELLIPSIS = 299,
     STAR = 300,
     _TRUE = 301,
     _FALSE = 302,
     DERIVED = 303,
     STRUCT = 304,
     INTERFACE = 305,
     CONNECTION = 306,
     PRENODE = 307,
     POSTNODE = 308,
     EXPECTS = 309,
     IMPLEMENTS = 310,
     SHARED = 311,
     INATTRPSET = 312,
     OUTATTRPSET = 313,
     PSET = 314,
     INITPHASE = 315,
     RUNTIMEPHASE = 316,
     FINALPHASE = 317,
     LOADPHASE = 318,
     CONSTANT = 319,
     VARIABLE = 320,
     USERFUNCTION = 321,
     PREDICATEFUNCTION = 322,
     INITIALIZE = 323,
     EXECUTE = 324,
     CATEGORY = 325,
     VOID = 326,
     PRE = 327,
     POST = 328,
     GRIDLAYERS = 329,
     THREADS = 330,
     OPTIONAL = 331,
     FRAMEWORK = 332
   };
#endif
/* Tokens.  */
#define DOUBLE_CONSTANT 258
#define INT_CONSTANT 259
#define STRING_LITERAL 260
#define IDENTIFIER 261
#define STRING 262
#define BOOL 263
#define CHAR 264
#define SHORT 265
#define INT 266
#define LONG 267
#define FLOAT 268
#define DOUBLE 269
#define UNSIGNED 270
#define EDGE 271
#define EDGESET 272
#define EDGETYPE 273
#define FUNCTOR 274
#define GRID 275
#define NODE 276
#define NODESET 277
#define NODETYPE 278
#define SERVICE 279
#define REPERTOIRE 280
#define TRIGGER 281
#define TRIGGEREDFUNCTION 282
#define SERIAL 283
#define PARALLEL 284
#define PARAMETERSET 285
#define NDPAIRLIST 286
#define OR 287
#define AND 288
#define EQUAL 289
#define NOT_EQUAL 290
#define LESS_EQUAL 291
#define GREATER_EQUAL 292
#define LESS 293
#define GREATER 294
#define DOT 295
#define AMPERSAND 296
#define LEFTSHIFT 297
#define RIGHTSHIFT 298
#define ELLIPSIS 299
#define STAR 300
#define _TRUE 301
#define _FALSE 302
#define DERIVED 303
#define STRUCT 304
#define INTERFACE 305
#define CONNECTION 306
#define PRENODE 307
#define POSTNODE 308
#define EXPECTS 309
#define IMPLEMENTS 310
#define SHARED 311
#define INATTRPSET 312
#define OUTATTRPSET 313
#define PSET 314
#define INITPHASE 315
#define RUNTIMEPHASE 316
#define FINALPHASE 317
#define LOADPHASE 318
#define CONSTANT 319
#define VARIABLE 320
#define USERFUNCTION 321
#define PREDICATEFUNCTION 322
#define INITIALIZE 323
#define EXECUTE 324
#define CATEGORY 325
#define VOID 326
#define PRE 327
#define POST 328
#define GRIDLAYERS 329
#define THREADS 330
#define OPTIONAL 331
#define FRAMEWORK 332




/* Copy the first part of user declarations.  */
#line 1 "mdl.y"

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
#line 18 "mdl.y"

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
#line 143 "mdl.y"

#ifndef YYSTYPE_DEFINITION
#define YYSTYPE_DEFINITION


/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 149 "mdl.y"
{
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
/* Line 193 of yacc.c.  */
#line 431 "mdl.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
} YYLTYPE;
# define yyltype YYLTYPE /* obsolescent; will be withdrawn */
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


/* Copy the second part of user declarations.  */
#line 191 "mdl.y"

#endif


/* Line 216 of yacc.c.  */
#line 459 "mdl.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
	     && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
    YYLTYPE yyls;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  29
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   2290

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  87
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  71
/* YYNRULES -- Number of rules.  */
#define YYNRULES  239
/* YYNRULES -- Number of states.  */
#define YYNSTATES  511

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   332

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      81,    82,     2,     2,    83,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    84,    80,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    85,     2,    86,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    78,     2,    79,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     7,    10,    12,    14,    16,    18,
      20,    22,    24,    26,    32,    39,    45,    53,    59,    61,
      64,    66,    68,    70,    72,    74,    76,    78,    80,    82,
      84,    86,    88,    90,    98,   104,   106,   109,   111,   113,
     115,   117,   119,   121,   123,   125,   127,   129,   131,   133,
     135,   137,   142,   147,   152,   155,   157,   160,   162,   164,
     166,   168,   170,   178,   184,   186,   189,   191,   193,   195,
     197,   199,   201,   203,   205,   207,   209,   211,   222,   233,
     244,   256,   258,   260,   262,   265,   267,   269,   280,   292,
     294,   296,   298,   300,   302,   305,   307,   309,   311,   313,
     320,   329,   336,   345,   353,   359,   361,   364,   366,   368,
     370,   372,   376,   381,   385,   387,   391,   393,   397,   399,
     403,   408,   410,   414,   418,   422,   426,   430,   434,   438,
     442,   446,   450,   454,   459,   463,   467,   471,   475,   479,
     483,   485,   487,   491,   498,   506,   515,   525,   529,   531,
     535,   539,   543,   545,   547,   549,   551,   553,   557,   561,
     565,   569,   573,   577,   581,   585,   587,   590,   595,   601,
     605,   610,   612,   617,   624,   628,   635,   644,   646,   650,
     652,   654,   660,   667,   675,   684,   686,   689,   691,   693,
     695,   701,   708,   715,   724,   726,   728,   733,   739,   745,
     753,   755,   759,   762,   766,   771,   776,   780,   785,   790,
     792,   794,   796,   799,   801,   804,   806,   808,   810,   812,
     814,   816,   818,   820,   823,   825,   827,   829,   831,   833,
     835,   837,   839,   841,   843,   845,   847,   849,   851,   853
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      88,     0,    -1,    89,    -1,    90,    -1,    89,    90,    -1,
      91,    -1,    92,    -1,    93,    -1,    96,    -1,   105,    -1,
     118,    -1,   144,    -1,     1,    -1,    49,     6,    78,   137,
      79,    -1,    77,    49,     6,    78,   137,    79,    -1,    50,
       6,    78,   137,    79,    -1,    16,     6,    55,   133,    78,
      94,    79,    -1,    16,     6,    78,    94,    79,    -1,    95,
      -1,    94,    95,    -1,    99,    -1,   139,    -1,   138,    -1,
     128,    -1,   134,    -1,   135,    -1,   108,    -1,   102,    -1,
     100,    -1,   101,    -1,   121,    -1,   123,    -1,   152,    -1,
      21,     6,    55,   133,    78,    97,    79,    -1,    21,     6,
      78,    97,    79,    -1,    98,    -1,    97,    98,    -1,    99,
      -1,   139,    -1,   138,    -1,   130,    -1,   134,    -1,   135,
      -1,   112,    -1,   102,    -1,   100,    -1,   101,    -1,   121,
      -1,   123,    -1,   152,    -1,    80,    -1,    57,    78,   137,
      79,    -1,    58,    78,   137,    79,    -1,    56,    78,   103,
      79,    -1,    56,   104,    -1,   104,    -1,   103,   104,    -1,
      99,    -1,   139,    -1,   131,    -1,   153,    -1,   138,    -1,
      65,     6,    55,   133,    78,   106,    79,    -1,    65,     6,
      78,   106,    79,    -1,   107,    -1,   106,   107,    -1,    99,
      -1,   139,    -1,   138,    -1,   129,    -1,   134,    -1,   112,
      -1,   100,    -1,   101,    -1,   121,    -1,   123,    -1,   152,
      -1,    51,    72,    21,    81,    82,    54,   133,    78,   110,
      79,    -1,    51,    73,    21,    81,    82,    54,   133,    78,
     110,    79,    -1,    51,    72,   109,    81,    82,    54,   133,
      78,   114,    79,    -1,    51,    72,   109,    81,   136,    82,
      54,   133,    78,   114,    79,    -1,    64,    -1,    65,    -1,
     111,    -1,   110,   111,    -1,    99,    -1,   116,    -1,    51,
      72,   113,    81,    82,    54,   133,    78,   114,    79,    -1,
      51,    72,   113,    81,   136,    82,    54,   133,    78,   114,
      79,    -1,    21,    -1,    16,    -1,    64,    -1,    65,    -1,
     115,    -1,   114,   115,    -1,    99,    -1,   116,    -1,   117,
      -1,   122,    -1,     6,    40,     6,    43,   125,    80,    -1,
       6,    40,     6,    43,    56,    40,   125,    80,    -1,    59,
      40,     6,    43,   125,    80,    -1,    59,    40,     6,    43,
      56,    40,   125,    80,    -1,    64,     6,    55,   133,    78,
     119,    79,    -1,    64,     6,    78,   119,    79,    -1,   120,
      -1,   119,   120,    -1,    99,    -1,   139,    -1,   134,    -1,
     101,    -1,    66,   124,    80,    -1,     6,    81,    82,    80,
      -1,    67,   124,    80,    -1,     6,    -1,   124,    83,     6,
      -1,     6,    -1,   125,    40,     6,    -1,     6,    -1,     6,
      81,    82,    -1,     6,    81,   124,    82,    -1,   126,    -1,
     127,    83,   126,    -1,    60,   127,    80,    -1,    61,   127,
      80,    -1,    62,   127,    80,    -1,    63,   127,    80,    -1,
      60,   127,    80,    -1,    61,   127,    80,    -1,    62,   127,
      80,    -1,    63,   127,    80,    -1,    60,   127,    80,    -1,
      61,   127,    80,    -1,    61,    74,   127,    80,    -1,    62,
     127,    80,    -1,    63,   127,    80,    -1,    60,   127,    80,
      -1,    61,   127,    80,    -1,    62,   127,    80,    -1,    63,
     127,    80,    -1,     6,    -1,   132,    -1,   133,    83,   132,
      -1,     6,    40,     6,    42,   125,    80,    -1,     6,    40,
       6,    42,    41,   125,    80,    -1,     6,    40,     6,    42,
      56,    40,   125,    80,    -1,     6,    40,     6,    42,    41,
      56,    40,   125,    80,    -1,    81,   136,    82,    -1,     6,
      -1,     6,    81,    82,    -1,    56,    40,     6,    -1,    59,
      40,     6,    -1,     5,    -1,     4,    -1,     3,    -1,    46,
      -1,    47,    -1,   136,    34,   136,    -1,   136,    35,   136,
      -1,   136,    36,   136,    -1,   136,    37,   136,    -1,   136,
      38,   136,    -1,   136,    39,   136,    -1,   136,    33,   136,
      -1,   136,    32,   136,    -1,   139,    -1,   137,   139,    -1,
      76,   154,   141,    80,    -1,    76,    48,   154,   141,    80,
      -1,   155,   141,    80,    -1,    48,   155,   141,    80,    -1,
       6,    -1,     6,    81,     4,    82,    -1,     6,    81,     4,
      83,     4,    82,    -1,     6,    84,     5,    -1,     6,    81,
       4,    82,    84,     5,    -1,     6,    81,     4,    83,     4,
      82,    84,     5,    -1,   140,    -1,   141,    83,   140,    -1,
       6,    -1,   142,    -1,    19,     6,    78,   145,    79,    -1,
      77,    19,     6,    78,   145,    79,    -1,    19,     6,    70,
       5,    78,   145,    79,    -1,    77,    19,     6,    70,     5,
      78,   145,    79,    -1,   146,    -1,   145,   146,    -1,    99,
      -1,   147,    -1,   149,    -1,   148,    69,    81,    82,    80,
      -1,   148,    69,    81,    44,    82,    80,    -1,   148,    69,
      81,   150,    82,    80,    -1,   148,    69,    81,   150,    83,
      44,    82,    80,    -1,    71,    -1,   155,    -1,    68,    81,
      82,    80,    -1,    68,    81,    44,    82,    80,    -1,    68,
      81,   150,    82,    80,    -1,    68,    81,   150,    83,    44,
      82,    80,    -1,   151,    -1,   150,    83,   151,    -1,   155,
     143,    -1,    27,   124,    80,    -1,    28,    27,   124,    80,
      -1,    29,    27,   124,    80,    -1,    27,   124,    80,    -1,
      28,    27,   124,    80,    -1,    29,    27,   124,    80,    -1,
     156,    -1,   157,    -1,   156,    -1,   156,    45,    -1,   157,
      -1,   157,    45,    -1,     7,    -1,     8,    -1,     9,    -1,
      10,    -1,    11,    -1,    12,    -1,    13,    -1,    14,    -1,
      12,    14,    -1,    15,    -1,    16,    -1,    17,    -1,    18,
      -1,    19,    -1,    20,    -1,    21,    -1,    22,    -1,    23,
      -1,    24,    -1,    25,    -1,    26,    -1,    30,    -1,    31,
      -1,     6,    -1,   155,    85,    86,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   348,   348,   353,   356,   361,   364,   367,   370,   373,
     376,   379,   390,   407,   412,   419,   426,   431,   438,   442,
     448,   452,   456,   460,   464,   468,   472,   476,   480,   484,
     488,   492,   496,   502,   507,   514,   518,   524,   528,   532,
     536,   540,   544,   548,   552,   556,   560,   564,   568,   572,
     578,   584,   590,   596,   600,   607,   611,   617,   621,   625,
     629,   633,   639,   644,   651,   655,   661,   665,   669,   673,
     677,   681,   685,   689,   693,   697,   701,   707,   711,   715,
     721,   729,   732,   738,   742,   748,   752,   758,   764,   772,
     775,   778,   781,   786,   790,   796,   800,   804,   808,   814,
     820,   828,   833,   840,   845,   852,   856,   862,   866,   870,
     874,   880,   886,   893,   899,   904,   911,   916,   923,   928,
     933,   940,   944,   950,   955,   960,   965,   972,   977,   982,
     987,   994,   999,  1004,  1009,  1014,  1021,  1026,  1031,  1036,
    1043,  1050,  1054,  1060,  1066,  1074,  1080,  1089,  1092,  1096,
    1100,  1104,  1108,  1114,  1119,  1124,  1127,  1130,  1133,  1136,
    1139,  1142,  1145,  1148,  1151,  1156,  1160,  1166,  1170,  1176,
    1180,  1186,  1191,  1196,  1201,  1207,  1213,  1221,  1225,  1231,
    1238,  1244,  1249,  1255,  1261,  1270,  1274,  1280,  1284,  1288,
    1294,  1298,  1302,  1306,  1312,  1316,  1322,  1326,  1330,  1334,
    1340,  1344,  1350,  1356,  1360,  1364,  1370,  1374,  1378,  1389,
    1393,  1399,  1403,  1407,  1411,  1417,  1421,  1425,  1429,  1433,
    1437,  1441,  1445,  1449,  1453,  1457,  1461,  1465,  1469,  1473,
    1477,  1481,  1485,  1489,  1493,  1497,  1501,  1505,  1509,  1516
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "DOUBLE_CONSTANT", "INT_CONSTANT",
  "STRING_LITERAL", "IDENTIFIER", "STRING", "BOOL", "CHAR", "SHORT", "INT",
  "LONG", "FLOAT", "DOUBLE", "UNSIGNED", "EDGE", "EDGESET", "EDGETYPE",
  "FUNCTOR", "GRID", "NODE", "NODESET", "NODETYPE", "SERVICE",
  "REPERTOIRE", "TRIGGER", "TRIGGEREDFUNCTION", "SERIAL", "PARALLEL",
  "PARAMETERSET", "NDPAIRLIST", "OR", "AND", "EQUAL", "NOT_EQUAL",
  "LESS_EQUAL", "GREATER_EQUAL", "LESS", "GREATER", "DOT", "AMPERSAND",
  "LEFTSHIFT", "RIGHTSHIFT", "ELLIPSIS", "STAR", "_TRUE", "_FALSE",
  "DERIVED", "STRUCT", "INTERFACE", "CONNECTION", "PRENODE", "POSTNODE",
  "EXPECTS", "IMPLEMENTS", "SHARED", "INATTRPSET", "OUTATTRPSET", "PSET",
  "INITPHASE", "RUNTIMEPHASE", "FINALPHASE", "LOADPHASE", "CONSTANT",
  "VARIABLE", "USERFUNCTION", "PREDICATEFUNCTION", "INITIALIZE", "EXECUTE",
  "CATEGORY", "VOID", "PRE", "POST", "GRIDLAYERS", "THREADS", "OPTIONAL",
  "FRAMEWORK", "'{'", "'}'", "';'", "'('", "')'", "','", "':'", "'['",
  "']'", "$accept", "mdlFile", "parserLineList", "parserLine", "struct",
  "interface", "edge", "edgeStatementList", "edgeStatement", "node",
  "nodeStatementList", "nodeStatement", "noop", "inAttrPSet",
  "outAttrPSet", "shared", "sharedStatementList", "sharedStatement",
  "variable", "variableStatementList", "variableStatement",
  "edgeConnection", "edgeConnectionComponentType",
  "edgeConnectionStatementList", "edgeConnectionStatement", "connection",
  "connectionComponentType", "connectionStatementList",
  "connectionStatement", "interfaceToMember", "psetToMember", "constant",
  "constantStatementList", "constantStatement", "userFunction",
  "userFunctionCall", "predicateFunction", "identifierList",
  "identifierDotList", "phaseIdentifier", "phaseIdentifierList",
  "edgeInstancePhase", "variableInstancePhase", "nodeInstancePhase",
  "sharedPhase", "interfacePointer", "interfacePointerList",
  "instanceMapping", "sharedMapping", "predicate", "dataTypeList",
  "optionalDataType", "dataType", "nameComment", "nameCommentList",
  "nameCommentArgument", "nameCommentArgumentList", "functor",
  "functorStatementList", "functorStatement", "execute", "returnType",
  "initialize", "argumentDataTypeList", "argumentDataType",
  "triggeredFunctionInstance", "triggeredFunctionShared",
  "nonPointerTypeClassifier", "typeClassifier", "typeCore", "array", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   123,   125,
      59,    40,    41,    44,    58,    91,    93
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    87,    88,    89,    89,    90,    90,    90,    90,    90,
      90,    90,    90,    91,    91,    92,    93,    93,    94,    94,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    96,    96,    97,    97,    98,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      99,   100,   101,   102,   102,   103,   103,   104,   104,   104,
     104,   104,   105,   105,   106,   106,   107,   107,   107,   107,
     107,   107,   107,   107,   107,   107,   107,   108,   108,   108,
     108,   109,   109,   110,   110,   111,   111,   112,   112,   113,
     113,   113,   113,   114,   114,   115,   115,   115,   115,   116,
     116,   117,   117,   118,   118,   119,   119,   120,   120,   120,
     120,   121,   122,   123,   124,   124,   125,   125,   126,   126,
     126,   127,   127,   128,   128,   128,   128,   129,   129,   129,
     129,   130,   130,   130,   130,   130,   131,   131,   131,   131,
     132,   133,   133,   134,   134,   135,   135,   136,   136,   136,
     136,   136,   136,   136,   136,   136,   136,   136,   136,   136,
     136,   136,   136,   136,   136,   137,   137,   138,   138,   139,
     139,   140,   140,   140,   140,   140,   140,   141,   141,   142,
     143,   144,   144,   144,   144,   145,   145,   146,   146,   146,
     147,   147,   147,   147,   148,   148,   149,   149,   149,   149,
     150,   150,   151,   152,   152,   152,   153,   153,   153,   154,
     154,   155,   155,   155,   155,   156,   156,   156,   156,   156,
     156,   156,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   156,   156,   156,   156,   156,   156,   156,   156,   157
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     2,     1,     1,     1,     1,     1,
       1,     1,     1,     5,     6,     5,     7,     5,     1,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     7,     5,     1,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     4,     4,     4,     2,     1,     2,     1,     1,     1,
       1,     1,     7,     5,     1,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,    10,    10,    10,
      11,     1,     1,     1,     2,     1,     1,    10,    11,     1,
       1,     1,     1,     1,     2,     1,     1,     1,     1,     6,
       8,     6,     8,     7,     5,     1,     2,     1,     1,     1,
       1,     3,     4,     3,     1,     3,     1,     3,     1,     3,
       4,     1,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     4,     3,     3,     3,     3,     3,     3,
       1,     1,     3,     6,     7,     8,     9,     3,     1,     3,
       3,     3,     1,     1,     1,     1,     1,     3,     3,     3,
       3,     3,     3,     3,     3,     1,     2,     4,     5,     3,
       4,     1,     4,     6,     3,     6,     8,     1,     3,     1,
       1,     5,     6,     7,     8,     1,     2,     1,     1,     1,
       5,     6,     6,     8,     1,     1,     4,     5,     5,     7,
       1,     3,     2,     3,     4,     4,     3,     4,     4,     1,
       1,     1,     2,     1,     2,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,    12,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     3,     5,     6,     7,     8,     9,    10,    11,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     1,
       4,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   140,   141,     0,   238,   215,
     216,   217,   218,   219,   220,   221,   222,   224,   225,   226,
     227,   228,   229,   230,   231,   232,   233,   234,   235,     0,
       0,     0,   236,   237,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,    18,    20,
      28,    29,    27,    26,    30,    31,    23,    24,    25,    22,
      21,    32,     0,   211,   213,     0,   238,     0,   194,   187,
       0,   185,   188,     0,   189,   195,     0,     0,     0,     0,
       0,     0,     0,    35,    37,    45,    46,    44,    43,    47,
      48,    40,    41,    42,    39,    38,    49,     0,   165,     0,
       0,   238,   107,   110,     0,   105,   109,   108,     0,     0,
       0,     0,     0,    66,    72,    73,     0,    64,    71,    74,
      75,    69,    70,    68,    67,    76,     0,     0,     0,     0,
       0,     0,   223,   114,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    57,    54,
      59,    61,    58,    60,     0,     0,   118,   121,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   209,   210,    17,
      19,   171,     0,   177,     0,   212,   214,     0,     0,   181,
     186,     0,     0,     0,     0,     0,     0,     0,     0,    34,
      36,    13,   166,    15,     0,     0,   104,   106,     0,     0,
       0,     0,     0,    63,    65,     0,     0,     0,     0,   142,
       0,   203,     0,     0,     0,     0,     0,    81,    82,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    55,
       0,     0,     0,   123,     0,   124,   125,   126,   111,   113,
       0,     0,     0,     0,   239,   169,     0,     0,     0,     0,
       0,   200,     0,     0,     0,    90,    89,    91,    92,     0,
     131,     0,   132,   134,   135,     0,     0,     0,   127,   128,
     129,   130,     0,   182,    14,    16,     0,   115,   204,   205,
     170,     0,     0,     0,   206,     0,     0,   136,   137,   138,
     139,    53,    56,    51,    52,   119,     0,   122,     0,   167,
       0,   174,   178,   183,     0,   196,     0,     0,   179,   180,
     202,     0,     0,     0,    33,     0,   133,   103,     0,    62,
       0,   116,     0,     0,     0,     0,   154,   153,   152,   148,
     155,   156,     0,     0,     0,     0,     0,     0,   207,   208,
     120,   168,   172,     0,   197,   198,     0,   201,     0,   190,
       0,     0,     0,     0,     0,   184,     0,     0,     0,     0,
     143,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     191,   192,     0,     0,     0,     0,   144,     0,   117,     0,
     149,   150,   151,   147,     0,   164,   163,   157,   158,   159,
     160,   161,   162,     0,     0,   175,   173,   199,     0,     0,
       0,     0,   145,     0,     0,     0,     0,     0,   193,     0,
       0,   146,     0,    85,     0,    83,    86,     0,     0,    95,
       0,    93,    96,    97,    98,     0,     0,   176,     0,     0,
       0,    77,    84,     0,     0,    79,    94,     0,    78,    87,
       0,     0,     0,     0,    80,    88,     0,   112,     0,     0,
       0,     0,     0,     0,    99,     0,   101,     0,     0,   100,
     102
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    10,    11,    12,    13,    14,    15,    87,    88,    16,
     122,   123,   109,    90,    91,    92,   268,   189,    17,   156,
     157,    93,   259,   464,   465,   128,   299,   470,   471,   472,
     473,    18,   144,   145,    94,   474,    95,   174,   364,   197,
     198,    96,   161,   131,   190,    46,    47,    97,    98,   376,
     137,    99,   138,   213,   214,   349,   350,    19,   110,   111,
     112,   113,   114,   290,   291,   101,   193,   205,   102,   103,
     104
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -404
static const yytype_int16 yypact[] =
{
     310,  -404,    66,   101,   128,   135,   169,   186,   195,    46,
      76,   320,  -404,  -404,  -404,  -404,  -404,  -404,  -404,  -404,
      14,   -20,    48,   141,   159,   125,   149,   233,   235,  -404,
    -404,   263,   849,   271,  1752,   263,   924,  2095,  2095,   263,
    1827,   263,  1149,   164,   200,  -404,  -404,   174,   246,  -404,
    -404,  -404,  -404,  -404,   270,  -404,  -404,  -404,  -404,  -404,
    -404,  -404,  -404,  -404,  -404,  -404,  -404,  -404,  -404,   289,
     296,   307,  -404,  -404,  2259,   293,  1224,   224,   238,   334,
     334,   334,   334,   289,   289,  2138,  -404,   549,  -404,  -404,
    -404,  -404,  -404,  -404,  -404,  -404,  -404,  -404,  -404,  -404,
    -404,  -404,     3,   299,   304,   278,  -404,   290,  -404,  -404,
    1449,  -404,  -404,   311,  -404,   288,   177,   324,   334,    28,
     334,   334,   624,  -404,  -404,  -404,  -404,  -404,  -404,  -404,
    -404,  -404,  -404,  -404,  -404,  -404,  -404,  1880,  -404,  1923,
     180,   362,  -404,  -404,  1475,  -404,  -404,  -404,   188,   334,
     334,   334,   334,  -404,  -404,  -404,   999,  -404,  -404,  -404,
    -404,  -404,  -404,  -404,  -404,  -404,   403,  1752,  2095,   849,
     263,   413,  -404,  -404,    13,   289,   289,     3,   150,   404,
     289,   448,   485,   334,   334,   334,   334,  1374,  -404,  -404,
    -404,  -404,  -404,  -404,  2095,  2095,   376,  -404,   127,   185,
     223,   274,   275,   312,  2259,   531,   288,    -5,    19,  -404,
    -404,    -6,   345,  -404,   335,  -404,  -404,  1752,   471,  -404,
    -404,   457,   924,   370,   340,   334,   350,   365,   373,  -404,
    -404,  -404,  -404,  -404,  1827,   545,  -404,  -404,  1149,   387,
     388,   389,   420,  -404,  -404,   474,  1550,  1966,   699,  -404,
     547,  -404,   588,   424,   425,   426,   520,  -404,  -404,   521,
     522,   430,   289,   289,   431,   459,   501,   502,  1299,  -404,
    2009,  2052,    15,  -404,   334,  -404,  -404,  -404,  -404,  -404,
     531,   503,   600,   603,  -404,  -404,   531,  1576,   532,   533,
     354,  -404,    29,   510,   774,  -404,  -404,  -404,  -404,   546,
    -404,   507,  -404,  -404,  -404,  1651,   584,  1074,  -404,  -404,
    -404,  -404,  1752,  -404,  -404,  -404,    43,  -404,  -404,  -404,
    -404,   574,   347,   575,  -404,   508,   515,  -404,  -404,  -404,
    -404,  -404,  -404,  -404,  -404,  -404,   367,  -404,   516,  -404,
     391,  -404,  -404,  -404,   578,  -404,   579,  2181,  -404,  -404,
    -404,   580,   581,   416,  -404,   358,  -404,  -404,    33,  -404,
    1726,  -404,    30,   620,    41,   609,  -404,  -404,  -404,   583,
    -404,  -404,   625,   626,   395,   614,   511,   615,  -404,  -404,
    -404,  -404,   586,   667,  -404,  -404,   591,  -404,   594,  -404,
     596,  2220,   623,   585,   672,  -404,   639,    51,   672,   677,
    -404,   263,   606,   683,   695,   660,   263,   395,   395,   395,
     395,   395,   395,   395,   395,   648,   263,   726,   650,   653,
    -404,  -404,   652,   263,   681,   672,  -404,    69,  -404,   209,
    -404,  -404,  -404,  -404,   216,  -404,  -404,  -404,  -404,  -404,
    -404,  -404,  -404,   263,   236,  -404,   654,  -404,   656,   244,
     263,    85,  -404,    31,    35,   252,    31,   732,  -404,    35,
     260,  -404,   700,  -404,    71,  -404,  -404,    -2,   701,  -404,
      39,  -404,  -404,  -404,  -404,    35,    77,  -404,    79,    35,
     733,  -404,  -404,   661,   738,  -404,  -404,   104,  -404,  -404,
     117,   702,   666,   705,  -404,  -404,    50,  -404,    61,   709,
      89,   711,   105,   672,  -404,   672,  -404,   132,   142,  -404,
    -404
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -404,  -404,  -404,   741,  -404,  -404,  -404,   589,   -75,  -404,
     541,  -116,   -32,    10,   -14,   -33,  -404,  -167,  -404,   526,
    -143,  -404,  -404,   297,  -403,   -40,  -404,  -178,  -252,  -265,
    -404,  -404,   534,  -126,    11,  -404,    18,    70,  -115,   480,
     227,  -404,  -404,  -404,  -404,   597,   -34,    -9,   -22,    52,
      -8,     6,   -17,   483,  -134,  -404,  -404,  -404,  -151,   -78,
    -404,  -404,  -404,   477,  -323,    21,  -404,   567,   -23,   -68,
     -56
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -214
static const yytype_int16 yytable[] =
{
      89,   116,   158,   127,   124,   140,   230,   148,   142,   211,
     153,   115,   210,   244,   133,   100,   246,   207,   237,   135,
     269,   173,   126,   147,   387,   164,   143,   132,   155,   208,
     139,   146,   220,   162,   196,   348,   361,   462,   480,   361,
     215,   467,   134,   255,   188,   467,   125,   129,   163,   361,
      33,   177,   154,   159,   130,    89,   361,   136,    34,   192,
     160,   482,   206,   165,   216,    27,   287,   361,   387,    31,
     100,   281,    20,   482,   394,   282,    29,   462,   283,   483,
    -211,   399,   191,   462,   362,   467,   396,   115,   212,   127,
     124,   399,    32,   251,   468,    28,   252,   335,   468,   363,
     133,   332,   225,    35,  -213,   135,   499,    21,   126,   399,
     467,    86,   142,   132,   212,    86,   158,   501,   485,    86,
     232,   400,   232,   467,   153,   399,    36,   147,   134,   399,
     143,   426,   125,   129,    22,   146,   207,    89,   468,   164,
     130,    23,   155,   136,   115,   399,   338,   162,   208,   452,
     481,    86,   100,   202,   203,   188,   488,    86,   489,    86,
     247,   360,   163,   468,   244,   461,   154,   159,   220,   504,
     192,   256,   399,   210,   160,    24,   468,   165,   230,   237,
      39,   206,   399,   494,    86,   506,   270,   271,   466,   127,
     124,   466,    25,   191,   115,   292,   495,    86,   158,   466,
     133,    26,   142,    40,    41,   135,   153,   273,   126,   220,
     274,   466,   509,   132,   257,   258,    89,   147,   486,    37,
     143,   164,   510,   115,   155,   146,   486,    42,   134,   162,
     232,   100,   125,   129,   166,   486,   188,    38,   486,    43,
     130,    44,   167,   136,   163,   253,   254,   397,   154,   159,
     261,   192,   169,   232,   232,   222,   160,   170,   234,   165,
     170,   127,   124,   170,   115,   275,   238,   158,   274,    45,
     292,   170,   133,   142,   191,   153,   105,   135,   168,   397,
     126,   478,   220,   427,   172,   132,   171,   453,   147,   115,
     164,   143,   170,   155,   454,   173,   146,   487,   162,   170,
     134,   490,   194,   276,   125,   129,   274,   199,   200,   201,
     451,     1,   130,   163,   456,   136,   195,   154,   159,   170,
      -2,     1,   459,   175,   292,   160,     2,   170,   165,     3,
     475,     4,   325,   326,   176,   170,     2,   115,   479,     3,
     196,     4,   336,   170,   215,   224,   226,   227,   228,   216,
     366,   367,   368,   369,   277,   278,   217,   274,   252,     5,
       6,   366,   367,   368,   369,   178,   179,   429,   292,     5,
       6,   218,   434,   212,     7,     8,   239,   240,   241,   242,
     221,   500,   444,   502,     7,     8,   295,     9,   507,   449,
     508,   296,   279,   370,   371,   252,   223,     9,   366,   367,
     368,   369,   235,   372,   370,   371,   373,   393,   245,   455,
     264,   265,   266,   267,   372,   285,   460,   373,   286,   250,
     300,   463,   469,   274,   463,   260,   405,   469,   374,   375,
     302,   284,   463,   274,   297,   298,   346,   347,   469,   374,
     392,   370,   371,   469,   463,   303,   469,   469,   274,   380,
     252,   372,   301,   304,   373,   469,   274,   272,   469,   435,
     436,   437,   438,   439,   440,   441,   442,   308,   309,   310,
     274,   274,   274,   382,   383,   262,   374,   106,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,   390,   391,
     311,    72,    73,   274,   318,   319,   320,   252,   252,   286,
     324,   327,   263,   252,   274,   288,   106,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,   211,   293,   328,
      72,    73,   274,   407,   408,   409,   410,   411,   412,   413,
     414,   306,   312,   289,   351,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,   329,   330,   339,   274,   274,   286,   356,   378,   316,
     274,   252,   352,   415,   317,   379,   381,    74,   252,   286,
      75,   321,   322,   323,   340,    76,    77,    78,   341,    79,
      80,    81,    82,   345,   344,    83,    84,   407,   408,   409,
     410,   411,   412,   413,   414,    85,   358,   355,   209,    86,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,   365,   377,   384,   385,
     398,   389,   388,   401,   402,   403,   404,   424,   406,   416,
     417,   418,    74,   419,   420,   117,   421,   423,   361,   425,
      76,    77,    78,   428,   118,   119,   120,   121,   430,   431,
      83,    84,   407,   408,   409,   410,   411,   412,   413,   414,
      85,   432,   443,   229,    86,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,   445,   446,   447,   448,   450,   458,   477,   457,   491,
     480,   484,   433,   492,   493,   496,   497,    74,   498,   503,
      75,   505,    30,   476,   337,    76,    77,    78,   248,    79,
      80,    81,    82,   294,   307,    83,    84,   249,   305,   342,
     353,   280,     0,     0,     0,    85,     0,     0,   315,    86,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    74,     0,     0,   117,     0,     0,     0,     0,
      76,    77,    78,     0,   118,   119,   120,   121,     0,     0,
      83,    84,     0,     0,     0,     0,     0,     0,     0,     0,
      85,     0,     0,   354,    86,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    74,     0,     0,
      75,     0,     0,     0,     0,    76,    77,    78,     0,    79,
      80,    81,    82,     0,     0,    83,    84,     0,     0,     0,
       0,     0,     0,     0,     0,    85,     0,     0,     0,    86,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    74,     0,     0,   117,     0,     0,     0,     0,
      76,    77,    78,     0,   118,   119,   120,   121,     0,     0,
      83,    84,     0,     0,     0,     0,     0,     0,     0,     0,
      85,     0,     0,     0,    86,   141,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    74,     0,     0,
     117,     0,     0,     0,     0,     0,    77,    78,     0,   149,
     150,   151,   152,     0,     0,    83,    84,     0,     0,     0,
       0,     0,     0,     0,     0,    85,     0,     0,   243,    86,
     141,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    74,     0,     0,   117,     0,     0,     0,     0,
       0,    77,    78,     0,   149,   150,   151,   152,     0,     0,
      83,    84,     0,     0,     0,     0,     0,     0,     0,     0,
      85,     0,     0,   359,    86,   141,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    74,     0,     0,
     117,     0,     0,     0,     0,     0,    77,    78,     0,   149,
     150,   151,   152,     0,     0,    83,    84,     0,     0,     0,
       0,     0,     0,     0,     0,    85,     0,     0,     0,    86,
     106,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,   180,   181,   182,    72,    73,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    74,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   183,   184,   185,   186,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      85,     0,   187,     0,    86,   106,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,   180,   181,   182,    72,
      73,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    74,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   183,
     184,   185,   186,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    85,     0,     0,   331,    86,
     106,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,   180,   181,   182,    72,    73,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    74,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   183,   184,   185,   186,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      85,     0,     0,     0,    86,   106,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,     0,     0,     0,    72,
      73,   141,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,     0,     0,     0,    72,    73,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   107,     0,     0,
     108,     0,     0,    74,     0,     0,     0,     0,   219,    86,
       0,     0,     0,    78,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   236,    86,   106,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,     0,     0,     0,
      72,    73,   106,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,     0,     0,     0,    72,    73,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   107,     0,
       0,   108,     0,     0,     0,     0,     0,     0,     0,   313,
      86,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   107,     0,     0,   108,     0,     0,
       0,     0,     0,     0,     0,   343,    86,   141,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,     0,     0,
       0,    72,    73,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    74,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    78,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     357,    86,   106,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,     0,     0,     0,    72,    73,   106,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,     0,
       0,     0,    72,    73,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   107,     0,     0,   108,     0,     0,
       0,     0,     0,     0,     0,   395,    86,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     107,     0,     0,   108,     0,     0,     0,     0,     0,     0,
       0,     0,    86,   141,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,     0,     0,     0,    72,    73,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    74,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    78,   106,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    86,     0,     0,
      72,    73,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    74,   106,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
       0,     0,     0,    72,    73,     0,     0,     0,     0,   231,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    74,   106,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,     0,     0,     0,    72,    73,     0,     0,
       0,     0,   233,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    74,   106,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,     0,     0,     0,    72,
      73,     0,     0,     0,     0,   314,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    74,   106,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,     0,
       0,     0,    72,    73,     0,     0,     0,     0,   333,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      74,   106,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,     0,     0,     0,    72,    73,     0,     0,     0,
       0,   334,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    74,   106,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,     0,     0,     0,    72,    73,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   204,   106,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,     0,     0,
       0,    72,    73,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   386,   106,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,     0,     0,     0,
      72,    73,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   422,   106,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,     0,     0,     0,    72,
      73
};

static const yytype_int16 yycheck[] =
{
      32,    35,    42,    36,    36,    39,   122,    41,    40,     6,
      42,    34,    87,   156,    36,    32,   167,    85,   144,    36,
     187,     6,    36,    40,   347,    42,    40,    36,    42,    85,
      38,    40,   110,    42,     6,     6,     6,     6,    40,     6,
      45,     6,    36,   177,    76,     6,    36,    36,    42,     6,
      70,    74,    42,    42,    36,    87,     6,    36,    78,    76,
      42,   464,    85,    42,    45,    19,   217,     6,   391,    55,
      87,   205,     6,   476,    41,    81,     0,     6,    84,    81,
      85,    40,    76,     6,    41,     6,    56,   110,    85,   122,
     122,    40,    78,    80,    59,    49,    83,    82,    59,    56,
     122,   268,    74,    55,    85,   122,    56,     6,   122,    40,
       6,    80,   144,   122,    85,    80,   156,    56,    79,    80,
     137,    80,   139,     6,   156,    40,    78,   144,   122,    40,
     144,    80,   122,   122,     6,   144,   204,   169,    59,   156,
     122,     6,   156,   122,   167,    40,   280,   156,   204,    80,
      79,    80,   169,    83,    84,   187,    79,    80,    79,    80,
     168,   312,   156,    59,   307,    80,   156,   156,   246,    80,
     187,    21,    40,   248,   156,     6,    59,   156,   294,   305,
      55,   204,    40,    79,    80,    80,   194,   195,   453,   222,
     222,   456,     6,   187,   217,   218,    79,    80,   238,   464,
     222,     6,   234,    78,    55,   222,   238,    80,   222,   287,
      83,   476,    80,   222,    64,    65,   248,   234,   470,    78,
     234,   238,    80,   246,   238,   234,   478,    78,   222,   238,
     247,   248,   222,   222,    70,   487,   268,    78,   490,     6,
     222,     6,    78,   222,   238,   175,   176,   362,   238,   238,
     180,   268,    78,   270,   271,    78,   238,    83,    78,   238,
      83,   294,   294,    83,   287,    80,    78,   307,    83,     6,
     293,    83,   294,   305,   268,   307,     5,   294,    78,   394,
     294,   459,   360,   398,    14,   294,    40,    78,   305,   312,
     307,   305,    83,   307,    78,     6,   305,   475,   307,    83,
     294,   479,    78,    80,   294,   294,    83,    80,    81,    82,
     425,     1,   294,   307,    78,   294,    78,   307,   307,    83,
       0,     1,    78,    27,   347,   307,    16,    83,   307,    19,
      78,    21,   262,   263,    27,    83,    16,   360,    78,    19,
       6,    21,   272,    83,    45,   118,   119,   120,   121,    45,
       3,     4,     5,     6,    80,    80,    78,    83,    83,    49,
      50,     3,     4,     5,     6,    72,    73,   401,   391,    49,
      50,    81,   406,    85,    64,    65,   149,   150,   151,   152,
      69,   496,   416,   498,    64,    65,    16,    77,   503,   423,
     505,    21,    80,    46,    47,    83,    72,    77,     3,     4,
       5,     6,    40,    56,    46,    47,    59,   355,     5,   443,
     183,   184,   185,   186,    56,    80,   450,    59,    83,     6,
      80,   453,   454,    83,   456,    21,   374,   459,    81,    82,
      80,    86,   464,    83,    64,    65,    82,    83,   470,    81,
      82,    46,    47,   475,   476,    80,   478,   479,    83,    82,
      83,    56,   225,    80,    59,   487,    83,    81,   490,   407,
     408,   409,   410,   411,   412,   413,   414,    80,    80,    80,
      83,    83,    83,    82,    83,    27,    81,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    82,    83,
      80,    30,    31,    83,    80,    80,    80,    83,    83,    83,
      80,    80,    27,    83,    83,    44,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,     6,    81,    80,
      30,    31,    83,    32,    33,    34,    35,    36,    37,    38,
      39,     6,    78,    82,    44,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    80,    80,    80,    83,    83,    83,    80,    80,    42,
      83,    83,    82,    82,     6,    80,    80,    48,    83,    83,
      51,    81,    81,    81,     4,    56,    57,    58,     5,    60,
      61,    62,    63,    80,    82,    66,    67,    32,    33,    34,
      35,    36,    37,    38,    39,    76,    42,    81,    79,    80,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    82,    82,    80,    80,
      40,    80,    82,    54,    81,    40,    40,    82,    54,    54,
      84,     4,    48,    82,    80,    51,    80,    54,     6,    40,
      56,    57,    58,     6,    60,    61,    62,    63,    82,     6,
      66,    67,    32,    33,    34,    35,    36,    37,    38,    39,
      76,     6,    54,    79,    80,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,     5,    82,    80,    82,    54,    80,     5,    84,     6,
      40,    40,    82,    82,     6,    43,    80,    48,    43,    40,
      51,    40,    11,   456,   274,    56,    57,    58,   169,    60,
      61,    62,    63,   222,   238,    66,    67,   170,   234,   286,
     293,   204,    -1,    -1,    -1,    76,    -1,    -1,    79,    80,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,
      56,    57,    58,    -1,    60,    61,    62,    63,    -1,    -1,
      66,    67,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    -1,    -1,    79,    80,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,    -1,    -1,
      51,    -1,    -1,    -1,    -1,    56,    57,    58,    -1,    60,
      61,    62,    63,    -1,    -1,    66,    67,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    80,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,
      56,    57,    58,    -1,    60,    61,    62,    63,    -1,    -1,
      66,    67,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    -1,    -1,    -1,    80,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,    -1,    -1,
      51,    -1,    -1,    -1,    -1,    -1,    57,    58,    -1,    60,
      61,    62,    63,    -1,    -1,    66,    67,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    79,    80,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,
      -1,    57,    58,    -1,    60,    61,    62,    63,    -1,    -1,
      66,    67,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    -1,    -1,    79,    80,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,    -1,    -1,
      51,    -1,    -1,    -1,    -1,    -1,    57,    58,    -1,    60,
      61,    62,    63,    -1,    -1,    66,    67,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    80,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    48,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    60,    61,    62,    63,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    -1,    78,    -1,    80,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    60,
      61,    62,    63,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    79,    80,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    48,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    60,    61,    62,    63,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    -1,    -1,    -1,    80,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    -1,    -1,    -1,    30,
      31,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    -1,    -1,    -1,    30,    31,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,    -1,
      71,    -1,    -1,    48,    -1,    -1,    -1,    -1,    79,    80,
      -1,    -1,    -1,    58,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    79,    80,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    -1,    -1,    -1,
      30,    31,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    -1,    -1,    -1,    30,    31,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,
      -1,    71,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    68,    -1,    -1,    71,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    -1,    -1,
      -1,    30,    31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    58,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    -1,    -1,    -1,    30,    31,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    -1,
      -1,    -1,    30,    31,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    68,    -1,    -1,    71,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      68,    -1,    -1,    71,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    80,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    -1,    -1,    -1,    30,    31,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    48,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    58,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    80,    -1,    -1,
      30,    31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      -1,    -1,    -1,    30,    31,    -1,    -1,    -1,    -1,    79,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    48,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    -1,    -1,    -1,    30,    31,    -1,    -1,
      -1,    -1,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    48,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    -1,    -1,    -1,    30,
      31,    -1,    -1,    -1,    -1,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    -1,
      -1,    -1,    30,    31,    -1,    -1,    -1,    -1,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      48,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    -1,    -1,    -1,    30,    31,    -1,    -1,    -1,
      -1,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    48,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    -1,    -1,    -1,    30,    31,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    48,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    -1,    -1,
      -1,    30,    31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    44,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    -1,    -1,    -1,
      30,    31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    44,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    -1,    -1,    -1,    30,
      31
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     1,    16,    19,    21,    49,    50,    64,    65,    77,
      88,    89,    90,    91,    92,    93,    96,   105,   118,   144,
       6,     6,     6,     6,     6,     6,     6,    19,    49,     0,
      90,    55,    78,    70,    78,    55,    78,    78,    78,    55,
      78,    55,    78,     6,     6,     6,   132,   133,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    48,    51,    56,    57,    58,    60,
      61,    62,    63,    66,    67,    76,    80,    94,    95,    99,
     100,   101,   102,   108,   121,   123,   128,   134,   135,   138,
     139,   152,   155,   156,   157,     5,     6,    68,    71,    99,
     145,   146,   147,   148,   149,   155,   133,    51,    60,    61,
      62,    63,    97,    98,    99,   100,   101,   102,   112,   121,
     123,   130,   134,   135,   138,   139,   152,   137,   139,   137,
     133,     6,    99,   101,   119,   120,   134,   139,   133,    60,
      61,    62,    63,    99,   100,   101,   106,   107,   112,   121,
     123,   129,   134,   138,   139,   152,    70,    78,    78,    78,
      83,    40,    14,     6,   124,    27,    27,   155,    72,    73,
      27,    28,    29,    60,    61,    62,    63,    78,    99,   104,
     131,   138,   139,   153,    78,    78,     6,   126,   127,   127,
     127,   127,   124,   124,    48,   154,   155,   156,   157,    79,
      95,     6,    85,   140,   141,    45,    45,    78,    81,    79,
     146,    69,    78,    72,   127,    74,   127,   127,   127,    79,
      98,    79,   139,    79,    78,    40,    79,   120,    78,   127,
     127,   127,   127,    79,   107,     5,   145,   137,    94,   132,
       6,    80,    83,   124,   124,   141,    21,    64,    65,   109,
      21,   124,    27,    27,   127,   127,   127,   127,   103,   104,
     137,   137,    81,    80,    83,    80,    80,    80,    80,    80,
     154,   141,    81,    84,    86,    80,    83,   145,    44,    82,
     150,   151,   155,    81,    97,    16,    21,    64,    65,   113,
      80,   127,    80,    80,    80,   119,     6,   106,    80,    80,
      80,    80,    78,    79,    79,    79,    42,     6,    80,    80,
      80,    81,    81,    81,    80,   124,   124,    80,    80,    80,
      80,    79,   104,    79,    79,    82,   124,   126,   141,    80,
       4,     5,   140,    79,    82,    80,    82,    83,     6,   142,
     143,    44,    82,   150,    79,    81,    80,    79,    42,    79,
     145,     6,    41,    56,   125,    82,     3,     4,     5,     6,
      46,    47,    56,    59,    81,    82,   136,    82,    80,    80,
      82,    80,    82,    83,    80,    80,    44,   151,    82,    80,
      82,    83,    82,   136,    41,    79,    56,   125,    40,    40,
      80,    54,    81,    40,    40,   136,    54,    32,    33,    34,
      35,    36,    37,    38,    39,    82,    54,    84,     4,    82,
      80,    80,    44,    54,    82,    40,    80,   125,     6,   133,
      82,     6,     6,    82,   133,   136,   136,   136,   136,   136,
     136,   136,   136,    54,   133,     5,    82,    80,    82,   133,
      54,   125,    80,    78,    78,   133,    78,    84,    80,    78,
     133,    80,     6,    99,   110,   111,   116,     6,    59,    99,
     114,   115,   116,   117,   122,    78,   110,     5,   114,    78,
      40,    79,   111,    81,    40,    79,   115,   114,    79,    79,
     114,     6,    82,     6,    79,    79,    43,    80,    43,    56,
     125,    56,   125,    40,    80,    40,    80,   125,   125,    80,
      80
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (&yylloc, YYPARSE_PARAM, YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, &yylloc, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval, &yylloc, YYLEX_PARAM)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value, Location, YYPARSE_PARAM); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, void * YYPARSE_PARAM)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp, YYPARSE_PARAM)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
    void * YYPARSE_PARAM;
#endif
{
  if (!yyvaluep)
    return;
  YYUSE (yylocationp);
  YYUSE (YYPARSE_PARAM);
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, void * YYPARSE_PARAM)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep, yylocationp, YYPARSE_PARAM)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
    void * YYPARSE_PARAM;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp, YYPARSE_PARAM);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule, void * YYPARSE_PARAM)
#else
static void
yy_reduce_print (yyvsp, yylsp, yyrule, YYPARSE_PARAM)
    YYSTYPE *yyvsp;
    YYLTYPE *yylsp;
    int yyrule;
    void * YYPARSE_PARAM;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       , &(yylsp[(yyi + 1) - (yynrhs)])		       , YYPARSE_PARAM);
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, yylsp, Rule, YYPARSE_PARAM); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp, void * YYPARSE_PARAM)
#else
static void
yydestruct (yymsg, yytype, yyvaluep, yylocationp, YYPARSE_PARAM)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
    YYLTYPE *yylocationp;
    void * YYPARSE_PARAM;
#endif
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  YYUSE (YYPARSE_PARAM);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void * YYPARSE_PARAM);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */






/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void * YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void * YYPARSE_PARAM;
#endif
#endif
{
  /* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;
/* Location data for the look-ahead symbol.  */
YYLTYPE yylloc;

  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;

  /* The location stack.  */
  YYLTYPE yylsa[YYINITDEPTH];
  YYLTYPE *yyls = yylsa;
  YYLTYPE *yylsp;
  /* The locations where the error started and ended.  */
  YYLTYPE yyerror_range[2];

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;
  yylsp = yyls;
#if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  /* Initialize the default location before parsing starts.  */
  yylloc.first_line   = yylloc.last_line   = 1;
  yylloc.first_column = yylloc.last_column = 0;
#endif

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;
	YYLTYPE *yyls1 = yyls;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yyls1, yysize * sizeof (*yylsp),
		    &yystacksize);
	yyls = yyls1;
	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);
	YYSTACK_RELOCATE (yyls);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;
  *++yylsp = yylloc;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location.  */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 348 "mdl.y"
    {

;}
    break;

  case 3:
#line 353 "mdl.y"
    {

;}
    break;

  case 4:
#line 356 "mdl.y"
    {

;}
    break;

  case 5:
#line 361 "mdl.y"
    {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[(1) - (1)].P_struct));
;}
    break;

  case 6:
#line 364 "mdl.y"
    {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[(1) - (1)].P_interface));
;}
    break;

  case 7:
#line 367 "mdl.y"
    {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[(1) - (1)].P_edge));
;}
    break;

  case 8:
#line 370 "mdl.y"
    {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[(1) - (1)].P_node));
;}
    break;

  case 9:
#line 373 "mdl.y"
    {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[(1) - (1)].P_variable));
;}
    break;

  case 10:
#line 376 "mdl.y"
    {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[(1) - (1)].P_constant));
;}
    break;

  case 11:
#line 379 "mdl.y"
    {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[(1) - (1)].P_functor));
;}
    break;

  case 12:
#line 390 "mdl.y"
    {
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
;}
    break;

  case 13:
#line 407 "mdl.y"
    {
   (yyval.P_struct) = new C_struct(*(yyvsp[(2) - (5)].P_string), (yyvsp[(4) - (5)].P_dataTypeList));
   (yyval.P_struct)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);
   delete (yyvsp[(2) - (5)].P_string);
;}
    break;

  case 14:
#line 412 "mdl.y"
    {
   (yyval.P_struct) = new C_struct(*(yyvsp[(3) - (6)].P_string), (yyvsp[(5) - (6)].P_dataTypeList), true);
   (yyval.P_struct)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (6)]).first_line);
   delete (yyvsp[(3) - (6)].P_string);
;}
    break;

  case 15:
#line 419 "mdl.y"
    {
   (yyval.P_interface) = new C_interface(*(yyvsp[(2) - (5)].P_string), (yyvsp[(4) - (5)].P_dataTypeList));
   (yyval.P_interface)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);
   delete (yyvsp[(2) - (5)].P_string);
;}
    break;

  case 16:
#line 426 "mdl.y"
    {
   (yyval.P_edge) = new C_edge(*(yyvsp[(2) - (7)].P_string), (yyvsp[(4) - (7)].P_interfacePointerList), (yyvsp[(6) - (7)].P_generalList));
   (yyval.P_edge)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (7)]).first_line);
   delete (yyvsp[(2) - (7)].P_string);
;}
    break;

  case 17:
#line 431 "mdl.y"
    {
   (yyval.P_edge) = new C_edge(*(yyvsp[(2) - (5)].P_string), 0, (yyvsp[(4) - (5)].P_generalList));
   (yyval.P_edge)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);
   delete (yyvsp[(2) - (5)].P_string);
;}
    break;

  case 18:
#line 438 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (1)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 19:
#line 442 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (2)].P_generalList), (yyvsp[(2) - (2)].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 20:
#line 448 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 21:
#line 452 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 22:
#line 456 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 23:
#line 460 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_phase);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 24:
#line 464 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_instanceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 25:
#line 468 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_sharedMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 26:
#line 472 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_connection);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 27:
#line 476 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_shared);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 28:
#line 480 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 29:
#line 484 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 30:
#line 488 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_userFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 31:
#line 492 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_predicateFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 32:
#line 496 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_triggeredFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 33:
#line 502 "mdl.y"
    {
   (yyval.P_node) = new C_node(*(yyvsp[(2) - (7)].P_string), (yyvsp[(4) - (7)].P_interfacePointerList), (yyvsp[(6) - (7)].P_generalList));
   (yyval.P_node)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (7)]).first_line);
   delete (yyvsp[(2) - (7)].P_string);
;}
    break;

  case 34:
#line 507 "mdl.y"
    {
   (yyval.P_node) = new C_node(*(yyvsp[(2) - (5)].P_string), 0, (yyvsp[(4) - (5)].P_generalList));
   (yyval.P_node)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);
   delete (yyvsp[(2) - (5)].P_string);
;}
    break;

  case 35:
#line 514 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (1)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 36:
#line 518 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (2)].P_generalList), (yyvsp[(2) - (2)].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 37:
#line 524 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 38:
#line 528 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 39:
#line 532 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 40:
#line 536 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_phase);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 41:
#line 540 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_instanceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 42:
#line 544 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_sharedMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 43:
#line 548 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_connection);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 44:
#line 552 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_shared);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 45:
#line 556 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 46:
#line 560 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 47:
#line 564 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_userFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 48:
#line 568 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_predicateFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 49:
#line 572 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_triggeredFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 50:
#line 578 "mdl.y"
    {
   (yyval.P_noop) = new C_noop();
   (yyval.P_noop)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 51:
#line 584 "mdl.y"
    {
   (yyval.P_general) = new C_inAttrPSet((yyvsp[(3) - (4)].P_dataTypeList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 52:
#line 590 "mdl.y"
    {
   (yyval.P_general) = new C_outAttrPSet((yyvsp[(3) - (4)].P_dataTypeList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 53:
#line 596 "mdl.y"
    {
   (yyval.P_shared) = new C_shared((yyvsp[(3) - (4)].P_generalList));
   (yyval.P_shared)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 54:
#line 600 "mdl.y"
    {
   (yyval.P_shared) = new C_shared();
   (yyval.P_shared)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
   (yyval.P_shared)->setGeneral((yyvsp[(2) - (2)].P_general));
;}
    break;

  case 55:
#line 607 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (1)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 56:
#line 611 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (2)].P_generalList), (yyvsp[(2) - (2)].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 57:
#line 617 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 58:
#line 621 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 59:
#line 625 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_phase);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 60:
#line 629 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_triggeredFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 61:
#line 633 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 62:
#line 639 "mdl.y"
    {
   (yyval.P_variable) = new C_variable(*(yyvsp[(2) - (7)].P_string), (yyvsp[(4) - (7)].P_interfacePointerList), (yyvsp[(6) - (7)].P_generalList));
   (yyval.P_variable)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (7)]).first_line);
   delete (yyvsp[(2) - (7)].P_string);
;}
    break;

  case 63:
#line 644 "mdl.y"
    {
   (yyval.P_variable) = new C_variable(*(yyvsp[(2) - (5)].P_string), 0, (yyvsp[(4) - (5)].P_generalList));
   (yyval.P_variable)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);
   delete (yyvsp[(2) - (5)].P_string);
;}
    break;

  case 64:
#line 651 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (1)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 65:
#line 655 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (2)].P_generalList), (yyvsp[(2) - (2)].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 66:
#line 661 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 67:
#line 665 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 68:
#line 669 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 69:
#line 673 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_phase);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 70:
#line 677 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_instanceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 71:
#line 681 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_connection);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 72:
#line 685 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 73:
#line 689 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 74:
#line 693 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_userFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 75:
#line 697 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_predicateFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 76:
#line 701 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_triggeredFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 77:
#line 707 "mdl.y"
    {
   (yyval.P_connection) = new C_edgeConnection((yyvsp[(7) - (10)].P_interfacePointerList), (yyvsp[(9) - (10)].P_generalList), Connection::_PRE);
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (10)]).first_line);
;}
    break;

  case 78:
#line 711 "mdl.y"
    {
   (yyval.P_connection) = new C_edgeConnection((yyvsp[(7) - (10)].P_interfacePointerList), (yyvsp[(9) - (10)].P_generalList), Connection::_POST);
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (10)]).first_line);
;}
    break;

  case 79:
#line 715 "mdl.y"
    {
   (yyval.P_connection) = new C_regularConnection((yyvsp[(7) - (10)].P_interfacePointerList), (yyvsp[(9) - (10)].P_generalList),
				(yyvsp[(3) - (10)].V_connectionComponentType),
				Connection::_PRE);
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (10)]).first_line);
;}
    break;

  case 80:
#line 721 "mdl.y"
    {
   (yyval.P_connection) = new C_regularConnection((yyvsp[(8) - (11)].P_interfacePointerList), (yyvsp[(10) - (11)].P_generalList),
				(yyvsp[(3) - (11)].V_connectionComponentType),
				Connection::_PRE, (yyvsp[(5) - (11)].P_predicate));
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (11)]).first_line);
;}
    break;

  case 81:
#line 729 "mdl.y"
    {
   (yyval.V_connectionComponentType) = Connection::_CONSTANT;
;}
    break;

  case 82:
#line 732 "mdl.y"
    {
   (yyval.V_connectionComponentType) = Connection::_VARIABLE;
;}
    break;

  case 83:
#line 738 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (1)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 84:
#line 742 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (2)].P_generalList), (yyvsp[(2) - (2)].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 85:
#line 748 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 86:
#line 752 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_interfaceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 87:
#line 758 "mdl.y"
    {
   (yyval.P_connection) = new C_regularConnection((yyvsp[(7) - (10)].P_interfacePointerList), (yyvsp[(9) - (10)].P_generalList),
				(yyvsp[(3) - (10)].V_connectionComponentType),
				Connection::_PRE);
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (10)]).first_line);
;}
    break;

  case 88:
#line 764 "mdl.y"
    {
   (yyval.P_connection) = new C_regularConnection((yyvsp[(8) - (11)].P_interfacePointerList), (yyvsp[(10) - (11)].P_generalList),
				(yyvsp[(3) - (11)].V_connectionComponentType),
				Connection::_PRE, (yyvsp[(5) - (11)].P_predicate));
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (11)]).first_line);
;}
    break;

  case 89:
#line 772 "mdl.y"
    {
   (yyval.V_connectionComponentType) = Connection::_NODE;
;}
    break;

  case 90:
#line 775 "mdl.y"
    {
   (yyval.V_connectionComponentType) = Connection::_EDGE;
;}
    break;

  case 91:
#line 778 "mdl.y"
    {
   (yyval.V_connectionComponentType) = Connection::_CONSTANT;
;}
    break;

  case 92:
#line 781 "mdl.y"
    {
   (yyval.V_connectionComponentType) = Connection::_VARIABLE;
;}
    break;

  case 93:
#line 786 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (1)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 94:
#line 790 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (2)].P_generalList), (yyvsp[(2) - (2)].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 95:
#line 796 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 96:
#line 800 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_interfaceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 97:
#line 804 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_psetMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 98:
#line 808 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_userFunctionCall);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 99:
#line 814 "mdl.y"
    {
   (yyval.P_interfaceMapping) = new C_interfaceToInstance(*(yyvsp[(1) - (6)].P_string), *(yyvsp[(3) - (6)].P_string), (yyvsp[(5) - (6)].P_identifierList));
   (yyval.P_interfaceMapping)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (6)]).first_line);
   delete (yyvsp[(1) - (6)].P_string);
   delete (yyvsp[(3) - (6)].P_string);
;}
    break;

  case 100:
#line 820 "mdl.y"
    {
   (yyval.P_interfaceMapping) = new C_interfaceToShared(*(yyvsp[(1) - (8)].P_string), *(yyvsp[(3) - (8)].P_string), (yyvsp[(7) - (8)].P_identifierList));
   (yyval.P_interfaceMapping)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (8)]).first_line);
   delete (yyvsp[(1) - (8)].P_string);
   delete (yyvsp[(3) - (8)].P_string);
;}
    break;

  case 101:
#line 828 "mdl.y"
    {
   (yyval.P_psetMapping) = new C_psetToInstance(*(yyvsp[(3) - (6)].P_string), (yyvsp[(5) - (6)].P_identifierList));
   (yyval.P_psetMapping)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (6)]).first_line);
   delete (yyvsp[(3) - (6)].P_string);
;}
    break;

  case 102:
#line 833 "mdl.y"
    {
   (yyval.P_psetMapping) = new C_psetToShared(*(yyvsp[(3) - (8)].P_string), (yyvsp[(7) - (8)].P_identifierList));
   (yyval.P_psetMapping)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (8)]).first_line);
   delete (yyvsp[(3) - (8)].P_string);
;}
    break;

  case 103:
#line 840 "mdl.y"
    {
   (yyval.P_constant) = new C_constant(*(yyvsp[(2) - (7)].P_string), (yyvsp[(4) - (7)].P_interfacePointerList), (yyvsp[(6) - (7)].P_generalList));
   (yyval.P_constant)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (7)]).first_line);
   delete (yyvsp[(2) - (7)].P_string);
;}
    break;

  case 104:
#line 845 "mdl.y"
    {
   (yyval.P_constant) = new C_constant(*(yyvsp[(2) - (5)].P_string), 0, (yyvsp[(4) - (5)].P_generalList));
   (yyval.P_constant)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);
   delete (yyvsp[(2) - (5)].P_string);
;}
    break;

  case 105:
#line 852 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (1)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 106:
#line 856 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (2)].P_generalList), (yyvsp[(2) - (2)].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 107:
#line 862 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 108:
#line 866 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 109:
#line 870 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_instanceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 110:
#line 874 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 111:
#line 880 "mdl.y"
    {
   (yyval.P_userFunction) = new C_userFunction((yyvsp[(2) - (3)].P_identifierList));
   (yyval.P_userFunction)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 112:
#line 886 "mdl.y"
    {
   (yyval.P_userFunctionCall) = new C_userFunctionCall(*(yyvsp[(1) - (4)].P_string));
   (yyval.P_userFunctionCall)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
   delete (yyvsp[(1) - (4)].P_string);
;}
    break;

  case 113:
#line 893 "mdl.y"
    {
   (yyval.P_predicateFunction) = new C_predicateFunction((yyvsp[(2) - (3)].P_identifierList));
   (yyval.P_predicateFunction)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 114:
#line 899 "mdl.y"
    {
   (yyval.P_identifierList) = new C_identifierList(*(yyvsp[(1) - (1)].P_string));
   (yyval.P_identifierList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
   delete (yyvsp[(1) - (1)].P_string);
;}
    break;

  case 115:
#line 904 "mdl.y"
    {
   (yyval.P_identifierList) = new C_identifierList((yyvsp[(1) - (3)].P_identifierList), *(yyvsp[(3) - (3)].P_string));
   (yyval.P_identifierList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
   delete (yyvsp[(3) - (3)].P_string);
;}
    break;

  case 116:
#line 911 "mdl.y"
    {
   (yyval.P_identifierList) = new C_identifierList(*(yyvsp[(1) - (1)].P_string));
   (yyval.P_identifierList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
   delete (yyvsp[(1) - (1)].P_string);
;}
    break;

  case 117:
#line 916 "mdl.y"
    {
   (yyval.P_identifierList) = new C_identifierList((yyvsp[(1) - (3)].P_identifierList), *(yyvsp[(3) - (3)].P_string));
   (yyval.P_identifierList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
   delete (yyvsp[(3) - (3)].P_string);
;}
    break;

  case 118:
#line 923 "mdl.y"
    {
   (yyval.P_phaseIdentifier) = new C_phaseIdentifier(*(yyvsp[(1) - (1)].P_string));
   (yyval.P_phaseIdentifier)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
   delete (yyvsp[(1) - (1)].P_string);
;}
    break;

  case 119:
#line 928 "mdl.y"
    {
   (yyval.P_phaseIdentifier) = new C_phaseIdentifier(*(yyvsp[(1) - (3)].P_string));
   (yyval.P_phaseIdentifier)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
   delete (yyvsp[(1) - (3)].P_string);
;}
    break;

  case 120:
#line 933 "mdl.y"
    {
   (yyval.P_phaseIdentifier) = new C_phaseIdentifier(*(yyvsp[(1) - (4)].P_string), (yyvsp[(3) - (4)].P_identifierList));
   (yyval.P_phaseIdentifier)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
   delete (yyvsp[(1) - (4)].P_string);
;}
    break;

  case 121:
#line 940 "mdl.y"
    {
   (yyval.P_phaseIdentifierList) = new C_phaseIdentifierList((yyvsp[(1) - (1)].P_phaseIdentifier));
   (yyval.P_phaseIdentifierList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 122:
#line 944 "mdl.y"
    {
   (yyval.P_phaseIdentifierList) = new C_phaseIdentifierList((yyvsp[(1) - (3)].P_phaseIdentifierList), (yyvsp[(3) - (3)].P_phaseIdentifier));
   (yyval.P_phaseIdentifierList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 123:
#line 950 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_initPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 124:
#line 955 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 125:
#line 960 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_finalPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 126:
#line 965 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_loadPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 127:
#line 972 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_initPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 128:
#line 977 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 129:
#line 982 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_finalPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 130:
#line 987 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_loadPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 131:
#line 994 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_initPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 132:
#line 999 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 133:
#line 1004 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeGridLayers());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[(3) - (4)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 134:
#line 1009 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_finalPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 135:
#line 1014 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_loadPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 136:
#line 1021 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   (yyval.P_phase) = new C_initPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 137:
#line 1026 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 138:
#line 1031 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   (yyval.P_phase) = new C_finalPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 139:
#line 1036 "mdl.y"
    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   (yyval.P_phase) = new C_loadPhase((yyvsp[(2) - (3)].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 140:
#line 1043 "mdl.y"
    {
   (yyval.P_interfacePointer) = new C_interfacePointer(*(yyvsp[(1) - (1)].P_string));
   (yyval.P_interfacePointer)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
   delete (yyvsp[(1) - (1)].P_string);
;}
    break;

  case 141:
#line 1050 "mdl.y"
    {
   (yyval.P_interfacePointerList) = new C_interfacePointerList((yyvsp[(1) - (1)].P_interfacePointer));
   (yyval.P_interfacePointerList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 142:
#line 1054 "mdl.y"
    {
   (yyval.P_interfacePointerList) = new C_interfacePointerList((yyvsp[(1) - (3)].P_interfacePointerList),(yyvsp[(3) - (3)].P_interfacePointer));
   (yyval.P_interfacePointerList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 143:
#line 1060 "mdl.y"
    {
   (yyval.P_instanceMapping) = new C_instanceMapping(*(yyvsp[(1) - (6)].P_string), *(yyvsp[(3) - (6)].P_string), (yyvsp[(5) - (6)].P_identifierList));
   (yyval.P_instanceMapping)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (6)]).first_line);
   delete (yyvsp[(1) - (6)].P_string);
   delete (yyvsp[(3) - (6)].P_string);
;}
    break;

  case 144:
#line 1066 "mdl.y"
    {
   (yyval.P_instanceMapping) = new C_instanceMapping(*(yyvsp[(1) - (7)].P_string), *(yyvsp[(3) - (7)].P_string), (yyvsp[(6) - (7)].P_identifierList), true);
   (yyval.P_instanceMapping)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (7)]).first_line);
   delete (yyvsp[(1) - (7)].P_string);
   delete (yyvsp[(3) - (7)].P_string);
;}
    break;

  case 145:
#line 1074 "mdl.y"
    {
   (yyval.P_sharedMapping) = new C_sharedMapping(*(yyvsp[(1) - (8)].P_string), *(yyvsp[(3) - (8)].P_string), (yyvsp[(7) - (8)].P_identifierList));
   (yyval.P_sharedMapping)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (8)]).first_line);
   delete (yyvsp[(1) - (8)].P_string);
   delete (yyvsp[(3) - (8)].P_string);
;}
    break;

  case 146:
#line 1080 "mdl.y"
    {
   (yyval.P_sharedMapping) = new C_sharedMapping(*(yyvsp[(1) - (9)].P_string), *(yyvsp[(3) - (9)].P_string), (yyvsp[(8) - (9)].P_identifierList), true);
   (yyval.P_sharedMapping)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (9)]).first_line);
   delete (yyvsp[(1) - (9)].P_string);
   delete (yyvsp[(3) - (9)].P_string);
;}
    break;

  case 147:
#line 1089 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new ParanthesisOp(), (yyvsp[(2) - (3)].P_predicate));  
;}
    break;

  case 148:
#line 1092 "mdl.y"
    {
   (yyval.P_predicate) = new InstancePredicate(new TerminalOp(), *(yyvsp[(1) - (1)].P_string));  
   delete (yyvsp[(1) - (1)].P_string);
;}
    break;

  case 149:
#line 1096 "mdl.y"
    {
   (yyval.P_predicate) = new FunctionPredicate(new TerminalOp(), *(yyvsp[(1) - (3)].P_string));  
   delete (yyvsp[(1) - (3)].P_string);
;}
    break;

  case 150:
#line 1100 "mdl.y"
    {
   (yyval.P_predicate) = new SharedPredicate(new TerminalOp(), *(yyvsp[(3) - (3)].P_string));  
   delete (yyvsp[(3) - (3)].P_string);
;}
    break;

  case 151:
#line 1104 "mdl.y"
    {
   (yyval.P_predicate) = new PSetPredicate(new TerminalOp(), *(yyvsp[(3) - (3)].P_string));  
   delete (yyvsp[(3) - (3)].P_string);
;}
    break;

  case 152:
#line 1108 "mdl.y"
    {
   std::ostringstream os;
   os << '"' << *(yyvsp[(1) - (1)].P_string) << '"';
   (yyval.P_predicate) = new Predicate(new TerminalOp(), os.str(), "string");  
   delete (yyvsp[(1) - (1)].P_string);
;}
    break;

  case 153:
#line 1114 "mdl.y"
    {
   std::ostringstream os;
   os << (yyvsp[(1) - (1)].V_int);
   (yyval.P_predicate) = new Predicate(new TerminalOp(), os.str(), "int");  
;}
    break;

  case 154:
#line 1119 "mdl.y"
    {
   std::ostringstream os;
   os << (yyvsp[(1) - (1)].V_double);
   (yyval.P_predicate) = new Predicate(new TerminalOp(), os.str(), "double");  
;}
    break;

  case 155:
#line 1124 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new TerminalOp(), "true", "bool");  
;}
    break;

  case 156:
#line 1127 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new TerminalOp(), "false", "bool");  
;}
    break;

  case 157:
#line 1130 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new EqualOp(), (yyvsp[(1) - (3)].P_predicate), (yyvsp[(3) - (3)].P_predicate));  
;}
    break;

  case 158:
#line 1133 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new NotEqualOp(), (yyvsp[(1) - (3)].P_predicate), (yyvsp[(3) - (3)].P_predicate));  
;}
    break;

  case 159:
#line 1136 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new LessEqualOp(), (yyvsp[(1) - (3)].P_predicate), (yyvsp[(3) - (3)].P_predicate));  
;}
    break;

  case 160:
#line 1139 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new GreaterEqualOp(), (yyvsp[(1) - (3)].P_predicate), (yyvsp[(3) - (3)].P_predicate));  
;}
    break;

  case 161:
#line 1142 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new LessOp(), (yyvsp[(1) - (3)].P_predicate), (yyvsp[(3) - (3)].P_predicate));  
;}
    break;

  case 162:
#line 1145 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new GreaterOp(), (yyvsp[(1) - (3)].P_predicate), (yyvsp[(3) - (3)].P_predicate));  
;}
    break;

  case 163:
#line 1148 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new AndOp(), (yyvsp[(1) - (3)].P_predicate), (yyvsp[(3) - (3)].P_predicate));  
;}
    break;

  case 164:
#line 1151 "mdl.y"
    {
   (yyval.P_predicate) = new Predicate(new OrOp(), (yyvsp[(1) - (3)].P_predicate), (yyvsp[(3) - (3)].P_predicate));  
;}
    break;

  case 165:
#line 1156 "mdl.y"
    {
   (yyval.P_dataTypeList) = new C_dataTypeList((yyvsp[(1) - (1)].P_dataType));
   (yyval.P_dataTypeList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 166:
#line 1160 "mdl.y"
    {
   (yyval.P_dataTypeList) = new C_dataTypeList((yyvsp[(1) - (2)].P_dataTypeList), (yyvsp[(2) - (2)].P_dataType));
   (yyval.P_dataTypeList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 167:
#line 1166 "mdl.y"
    {
   (yyval.P_dataType) = new C_dataType((yyvsp[(2) - (4)].P_typeClassifier), (yyvsp[(3) - (4)].P_nameCommentList), false, true);
   (yyval.P_dataType)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 168:
#line 1170 "mdl.y"
    {
   (yyval.P_dataType) = new C_dataType((yyvsp[(3) - (5)].P_typeClassifier), (yyvsp[(4) - (5)].P_nameCommentList), true, true);
   (yyval.P_dataType)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);
;}
    break;

  case 169:
#line 1176 "mdl.y"
    {
   (yyval.P_dataType) = new C_dataType((yyvsp[(1) - (3)].P_typeClassifier), (yyvsp[(2) - (3)].P_nameCommentList));
   (yyval.P_dataType)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 170:
#line 1180 "mdl.y"
    {
   (yyval.P_dataType) = new C_dataType((yyvsp[(2) - (4)].P_typeClassifier), (yyvsp[(3) - (4)].P_nameCommentList), true);
   (yyval.P_dataType)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 171:
#line 1186 "mdl.y"
    {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[(1) - (1)].P_string));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
   delete (yyvsp[(1) - (1)].P_string);
;}
    break;

  case 172:
#line 1191 "mdl.y"
    {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[(1) - (4)].P_string), (yyvsp[(3) - (4)].V_int));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
   delete (yyvsp[(1) - (4)].P_string);
;}
    break;

  case 173:
#line 1196 "mdl.y"
    {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[(1) - (6)].P_string), (yyvsp[(3) - (6)].V_int), (yyvsp[(5) - (6)].V_int));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (6)]).first_line);
   delete (yyvsp[(1) - (6)].P_string);
;}
    break;

  case 174:
#line 1201 "mdl.y"
    {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[(1) - (3)].P_string), *(yyvsp[(3) - (3)].P_string));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
   delete (yyvsp[(1) - (3)].P_string);
   delete (yyvsp[(3) - (3)].P_string);
;}
    break;

  case 175:
#line 1207 "mdl.y"
    {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[(1) - (6)].P_string), *(yyvsp[(6) - (6)].P_string), (yyvsp[(3) - (6)].V_int));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (6)]).first_line);
   delete (yyvsp[(1) - (6)].P_string);
   delete (yyvsp[(6) - (6)].P_string);
;}
    break;

  case 176:
#line 1213 "mdl.y"
    {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[(1) - (8)].P_string), *(yyvsp[(8) - (8)].P_string), (yyvsp[(3) - (8)].V_int), (yyvsp[(5) - (8)].V_int));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (8)]).first_line);
   delete (yyvsp[(1) - (8)].P_string);
   delete (yyvsp[(8) - (8)].P_string);
;}
    break;

  case 177:
#line 1221 "mdl.y"
    {   
   (yyval.P_nameCommentList) = new C_nameCommentList((yyvsp[(1) - (1)].P_nameComment));
   (yyval.P_nameCommentList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 178:
#line 1225 "mdl.y"
    {
   (yyval.P_nameCommentList) = new C_nameCommentList((yyvsp[(1) - (3)].P_nameCommentList), (yyvsp[(3) - (3)].P_nameComment));   
   (yyval.P_nameCommentList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 179:
#line 1231 "mdl.y"
    {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[(1) - (1)].P_string));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
   delete (yyvsp[(1) - (1)].P_string);
;}
    break;

  case 180:
#line 1238 "mdl.y"
    {
   (yyval.P_nameCommentList) = new C_nameCommentList((yyvsp[(1) - (1)].P_nameComment));
   (yyval.P_nameCommentList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 181:
#line 1244 "mdl.y"
    {
    (yyval.P_functor) = new C_functor(*(yyvsp[(2) - (5)].P_string), (yyvsp[(4) - (5)].P_generalList));
    (yyval.P_functor)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);
    delete (yyvsp[(2) - (5)].P_string);
;}
    break;

  case 182:
#line 1249 "mdl.y"
    {
    (yyval.P_functor) = new C_functor(*(yyvsp[(3) - (6)].P_string), (yyvsp[(5) - (6)].P_generalList));
    (yyval.P_functor)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (6)]).first_line);
    (yyval.P_functor)->setFrameWorkElement();
    delete (yyvsp[(3) - (6)].P_string);
;}
    break;

  case 183:
#line 1255 "mdl.y"
    {
    (yyval.P_functor) = new C_functor(*(yyvsp[(2) - (7)].P_string), (yyvsp[(6) - (7)].P_generalList), *(yyvsp[(4) - (7)].P_string));
    (yyval.P_functor)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (7)]).first_line);
    delete (yyvsp[(2) - (7)].P_string);
    delete (yyvsp[(4) - (7)].P_string);
;}
    break;

  case 184:
#line 1261 "mdl.y"
    {
    (yyval.P_functor) = new C_functor(*(yyvsp[(3) - (8)].P_string), (yyvsp[(7) - (8)].P_generalList), *(yyvsp[(5) - (8)].P_string));
    (yyval.P_functor)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (8)]).first_line);
    (yyval.P_functor)->setFrameWorkElement();
    delete (yyvsp[(3) - (8)].P_string);
    delete (yyvsp[(5) - (8)].P_string);
;}
    break;

  case 185:
#line 1270 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (1)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 186:
#line 1274 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (2)].P_generalList), (yyvsp[(2) - (2)].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 187:
#line 1280 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 188:
#line 1284 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 189:
#line 1288 "mdl.y"
    {
   (yyval.P_general) = (yyvsp[(1) - (1)].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 190:
#line 1294 "mdl.y"
    {
   (yyval.P_general) = new C_execute((yyvsp[(1) - (5)].P_returnType));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);  
;}
    break;

  case 191:
#line 1298 "mdl.y"
    {
   (yyval.P_general) = new C_execute((yyvsp[(1) - (6)].P_returnType), true);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (6)]).first_line);  
;}
    break;

  case 192:
#line 1302 "mdl.y"
    {
   (yyval.P_general) = new C_execute((yyvsp[(1) - (6)].P_returnType), (yyvsp[(4) - (6)].P_generalList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (6)]).first_line);  
;}
    break;

  case 193:
#line 1306 "mdl.y"
    {
   (yyval.P_general) = new C_execute((yyvsp[(1) - (8)].P_returnType), (yyvsp[(4) - (8)].P_generalList), true);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (8)]).first_line);  
;}
    break;

  case 194:
#line 1312 "mdl.y"
    {
   (yyval.P_returnType) = new C_returnType(true);
   (yyval.P_returnType)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);    
;}
    break;

  case 195:
#line 1316 "mdl.y"
    {
   (yyval.P_returnType) = new C_returnType((yyvsp[(1) - (1)].P_typeClassifier));
   (yyval.P_returnType)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);    
;}
    break;

  case 196:
#line 1322 "mdl.y"
    {
   (yyval.P_general) = new C_initialize();
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);  
;}
    break;

  case 197:
#line 1326 "mdl.y"
    {
   (yyval.P_general) = new C_initialize(true);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);  
;}
    break;

  case 198:
#line 1330 "mdl.y"
    {
   (yyval.P_general) = new C_initialize((yyvsp[(3) - (5)].P_generalList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (5)]).first_line);  
;}
    break;

  case 199:
#line 1334 "mdl.y"
    {
   (yyval.P_general) = new C_initialize((yyvsp[(3) - (7)].P_generalList), true);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (7)]).first_line);  
;}
    break;

  case 200:
#line 1340 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (1)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 201:
#line 1344 "mdl.y"
    {
   (yyval.P_generalList) = new C_generalList((yyvsp[(1) - (3)].P_generalList), (yyvsp[(3) - (3)].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 202:
#line 1350 "mdl.y"
    {
   (yyval.P_general) = new C_dataType((yyvsp[(1) - (2)].P_typeClassifier), (yyvsp[(2) - (2)].P_nameCommentList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 203:
#line 1356 "mdl.y"
    {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionInstance((yyvsp[(2) - (3)].P_identifierList), TriggeredFunction::_SERIAL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 204:
#line 1360 "mdl.y"
    {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionInstance((yyvsp[(3) - (4)].P_identifierList), TriggeredFunction::_SERIAL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 205:
#line 1364 "mdl.y"
    {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionInstance((yyvsp[(3) - (4)].P_identifierList), TriggeredFunction::_PARALLEL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 206:
#line 1370 "mdl.y"
    {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionShared((yyvsp[(2) - (3)].P_identifierList), TriggeredFunction::_SERIAL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;

  case 207:
#line 1374 "mdl.y"
    {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionShared((yyvsp[(3) - (4)].P_identifierList), TriggeredFunction::_SERIAL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 208:
#line 1378 "mdl.y"
    {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionShared((yyvsp[(3) - (4)].P_identifierList), TriggeredFunction::_PARALLEL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (4)]).first_line);
;}
    break;

  case 209:
#line 1389 "mdl.y"
    {
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[(1) - (1)].P_typeCore));
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 210:
#line 1393 "mdl.y"
    { 
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[(1) - (1)].P_array));
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 211:
#line 1399 "mdl.y"
    {
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[(1) - (1)].P_typeCore));
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 212:
#line 1403 "mdl.y"
    {
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[(1) - (2)].P_typeCore), true);
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 213:
#line 1407 "mdl.y"
    { 
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[(1) - (1)].P_array));
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 214:
#line 1411 "mdl.y"
    {
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[(1) - (2)].P_array), true);
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 215:
#line 1417 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new StringType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 216:
#line 1421 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new BoolType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 217:
#line 1425 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new CharType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 218:
#line 1429 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new ShortType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 219:
#line 1433 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new IntType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 220:
#line 1437 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new LongType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 221:
#line 1441 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new FloatType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 222:
#line 1445 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new DoubleType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 223:
#line 1449 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new LongDoubleType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (2)]).first_line);
;}
    break;

  case 224:
#line 1453 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new UnsignedType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 225:
#line 1457 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new EdgeType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 226:
#line 1461 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new EdgeSetType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 227:
#line 1465 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new EdgeTypeType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 228:
#line 1469 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new FunctorType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 229:
#line 1473 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new GridType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 230:
#line 1477 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new NodeType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 231:
#line 1481 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new NodeSetType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 232:
#line 1485 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new NodeTypeType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 233:
#line 1489 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new ServiceType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 234:
#line 1493 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new RepertoireType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 235:
#line 1497 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new TriggerType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 236:
#line 1501 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new ParameterSetType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 237:
#line 1505 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(new NDPairListType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
;}
    break;

  case 238:
#line 1509 "mdl.y"
    {
   (yyval.P_typeCore) = new C_typeCore(*(yyvsp[(1) - (1)].P_string));
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (1)]).first_line);
   delete (yyvsp[(1) - (1)].P_string);
;}
    break;

  case 239:
#line 1516 "mdl.y"
    {
   (yyval.P_array) = new C_array((yyvsp[(1) - (3)].P_typeClassifier));
   (yyval.P_array)->setTokenLocation(CURRENTFILE, (yylsp[(1) - (3)]).first_line);
;}
    break;


/* Line 1267 of yacc.c.  */
#line 4487 "mdl.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (&yylloc, YYPARSE_PARAM, YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (&yylloc, YYPARSE_PARAM, yymsg);
	  }
	else
	  {
	    yyerror (&yylloc, YYPARSE_PARAM, YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }

  yyerror_range[0] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval, &yylloc, YYPARSE_PARAM);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  yyerror_range[0] = yylsp[1-yylen];
  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;

      yyerror_range[0] = *yylsp;
      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp, yylsp, YYPARSE_PARAM);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;

  yyerror_range[1] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the look-ahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, (yyerror_range - 1), 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (&yylloc, YYPARSE_PARAM, YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval, &yylloc, YYPARSE_PARAM);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp, yylsp, YYPARSE_PARAM);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 1522 "mdl.y"

 
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

