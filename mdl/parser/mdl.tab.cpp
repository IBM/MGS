/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 2

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Substitute the type names.  */
#define YYSTYPE         MDLSTYPE
#define YYLTYPE         MDLLTYPE
/* Substitute the variable and function names.  */
#define yyparse         mdlparse
#define yylex           mdllex
#define yyerror         mdlerror
#define yydebug         mdldebug
#define yynerrs         mdlnerrs

/* First part of user prologue.  */
#line 1 "mdl.y"

// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "bison_compat.h"  // adds compatibility macros
#include "mdl_types.h"
#include "ParserClasses.h"
#include "MdlLexer.h"
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
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

using namespace std;

#ifndef USABLE
#define USABLE
#endif

// Define context access
#define CONTEXT ((MdlContext *) parm)
#define CURRENTFILE (CONTEXT->_lexer)->currentFileName
#define CURRENTLINE (CONTEXT->_lexer)->lineCount

// Function declarations
void mdlerror(const char *s);
void mdlerror(YYLTYPE* locp, void* parm, const char* s);
int mdllex(YYSTYPE* lvalp, YYLTYPE* locp, void* parm);

// Helper function for executing productions
inline void HIGH_LEVEL_EXECUTE(void* parm, C_production* l) {
    try {
        l->execute(CONTEXT);
        delete l;
        CONTEXT->setErrorDisplayed(false);
    } catch (SyntaxErrorException& e) {
        cerr << "ERROR: [";
        if (e.isCaught()) {
            cerr << e.getFileName() << ":" << e.getLineNumber();
        } else {
            // Use the context's stored error information
            CONTEXT->synchronize(); // Ensure context is synced before reporting
            cerr << CONTEXT->getLastErrorFileName() << ":" << CONTEXT->getLastErrorLine();
        }
        cerr << "] " << e.getError() << endl;
        CONTEXT->setError();
        delete l;
    } catch (InternalException& e) {
        cerr << "ERROR: [";
        CONTEXT->synchronize(); // Ensure context is synched before reporting
        cerr << CONTEXT->getLastErrorFileName() << ":" << CONTEXT->getLastErrorLine() << "] ";
        cerr << e.getError() << endl;
        CONTEXT->setError();
        delete l;
    }
}

#line 131 "mdl.y"

#ifndef YYSTYPE_DEFINITION
#define YYSTYPE_DEFINITION

#line 201 "mdl.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "mdl.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  symbol_YYEMPTY = -2,
  symbol_YYEOF = 0,                        /* "end of file"  */
  symbol_YYerror = 1,                      /* error  */
  symbol_YYUNDEF = 2,                      /* "invalid token"  */
  symbol_DOUBLE_CONSTANT = 3,              /* DOUBLE_CONSTANT  */
  symbol_INT_CONSTANT = 4,                 /* INT_CONSTANT  */
  symbol_STRING_LITERAL = 5,               /* STRING_LITERAL  */
  symbol_IDENTIFIER = 6,                   /* IDENTIFIER  */
  symbol_STRING = 7,                       /* STRING  */
  symbol_BOOL = 8,                         /* BOOL  */
  symbol_CHAR = 9,                         /* CHAR  */
  symbol_SHORT = 10,                       /* SHORT  */
  symbol_INT = 11,                         /* INT  */
  symbol_LONG = 12,                        /* LONG  */
  symbol_FLOAT = 13,                       /* FLOAT  */
  symbol_DOUBLE = 14,                      /* DOUBLE  */
  symbol_UNSIGNED = 15,                    /* UNSIGNED  */
  symbol_EDGE = 16,                        /* EDGE  */
  symbol_EDGESET = 17,                     /* EDGESET  */
  symbol_EDGETYPE = 18,                    /* EDGETYPE  */
  symbol_FUNCTOR = 19,                     /* FUNCTOR  */
  symbol_GRID = 20,                        /* GRID  */
  symbol_NODE = 21,                        /* NODE  */
  symbol_NODESET = 22,                     /* NODESET  */
  symbol_NODETYPE = 23,                    /* NODETYPE  */
  symbol_SERVICE = 24,                     /* SERVICE  */
  symbol_REPERTOIRE = 25,                  /* REPERTOIRE  */
  symbol_TRIGGER = 26,                     /* TRIGGER  */
  symbol_TRIGGEREDFUNCTION = 27,           /* TRIGGEREDFUNCTION  */
  symbol_SERIAL = 28,                      /* SERIAL  */
  symbol_PARALLEL = 29,                    /* PARALLEL  */
  symbol_PARAMETERSET = 30,                /* PARAMETERSET  */
  symbol_NDPAIRLIST = 31,                  /* NDPAIRLIST  */
  symbol_OR = 32,                          /* OR  */
  symbol_AND = 33,                         /* AND  */
  symbol_EQUAL = 34,                       /* EQUAL  */
  symbol_NOT_EQUAL = 35,                   /* NOT_EQUAL  */
  symbol_LESS_EQUAL = 36,                  /* LESS_EQUAL  */
  symbol_GREATER_EQUAL = 37,               /* GREATER_EQUAL  */
  symbol_LESS = 38,                        /* LESS  */
  symbol_GREATER = 39,                     /* GREATER  */
  symbol_DOT = 40,                         /* DOT  */
  symbol_AMPERSAND = 41,                   /* AMPERSAND  */
  symbol_LEFTSHIFT = 42,                   /* LEFTSHIFT  */
  symbol_RIGHTSHIFT = 43,                  /* RIGHTSHIFT  */
  symbol_ELLIPSIS = 44,                    /* ELLIPSIS  */
  symbol_STAR = 45,                        /* STAR  */
  symbol__TRUE = 46,                       /* _TRUE  */
  symbol__FALSE = 47,                      /* _FALSE  */
  symbol_DERIVED = 48,                     /* DERIVED  */
  symbol_STRUCT = 49,                      /* STRUCT  */
  symbol_INTERFACE = 50,                   /* INTERFACE  */
  symbol_CONNECTION = 51,                  /* CONNECTION  */
  symbol_PRENODE = 52,                     /* PRENODE  */
  symbol_POSTNODE = 53,                    /* POSTNODE  */
  symbol_EXPECTS = 54,                     /* EXPECTS  */
  symbol_IMPLEMENTS = 55,                  /* IMPLEMENTS  */
  symbol_SHARED = 56,                      /* SHARED  */
  symbol_INATTRPSET = 57,                  /* INATTRPSET  */
  symbol_OUTATTRPSET = 58,                 /* OUTATTRPSET  */
  symbol_PSET = 59,                        /* PSET  */
  symbol_INITPHASE = 60,                   /* INITPHASE  */
  symbol_RUNTIMEPHASE = 61,                /* RUNTIMEPHASE  */
  symbol_FINALPHASE = 62,                  /* FINALPHASE  */
  symbol_LOADPHASE = 63,                   /* LOADPHASE  */
  symbol_CONSTANT = 64,                    /* CONSTANT  */
  symbol_VARIABLE = 65,                    /* VARIABLE  */
  symbol_USERFUNCTION = 66,                /* USERFUNCTION  */
  symbol_PREDICATEFUNCTION = 67,           /* PREDICATEFUNCTION  */
  symbol_INITIALIZE = 68,                  /* INITIALIZE  */
  symbol_EXECUTE = 69,                     /* EXECUTE  */
  symbol_CATEGORY = 70,                    /* CATEGORY  */
  symbol_VOID = 71,                        /* VOID  */
  symbol_PRE = 72,                         /* PRE  */
  symbol_POST = 73,                        /* POST  */
  symbol_GRIDLAYERS = 74,                  /* GRIDLAYERS  */
  symbol_THREADS = 75,                     /* THREADS  */
  symbol_OPTIONAL = 76,                    /* OPTIONAL  */
  symbol_FRAMEWORK = 77,                   /* FRAMEWORK  */
  symbol_78_ = 78,                         /* '{'  */
  symbol_79_ = 79,                         /* '}'  */
  symbol_80_ = 80,                         /* ';'  */
  symbol_81_ = 81,                         /* '('  */
  symbol_82_ = 82,                         /* ')'  */
  symbol_83_ = 83,                         /* ','  */
  symbol_84_ = 84,                         /* ':'  */
  symbol_85_ = 85,                         /* '['  */
  symbol_86_ = 86,                         /* ']'  */
  symbol_YYACCEPT = 87,                    /* $accept  */
  symbol_mdlFile = 88,                     /* mdlFile  */
  symbol_parserLineList = 89,              /* parserLineList  */
  symbol_parserLine = 90,                  /* parserLine  */
  symbol_struct = 91,                      /* struct  */
  symbol_interface = 92,                   /* interface  */
  symbol_edge = 93,                        /* edge  */
  symbol_edgeStatementList = 94,           /* edgeStatementList  */
  symbol_edgeStatement = 95,               /* edgeStatement  */
  symbol_node = 96,                        /* node  */
  symbol_nodeStatementList = 97,           /* nodeStatementList  */
  symbol_nodeStatement = 98,               /* nodeStatement  */
  symbol_noop = 99,                        /* noop  */
  symbol_inAttrPSet = 100,                 /* inAttrPSet  */
  symbol_outAttrPSet = 101,                /* outAttrPSet  */
  symbol_shared = 102,                     /* shared  */
  symbol_sharedStatementList = 103,        /* sharedStatementList  */
  symbol_sharedStatement = 104,            /* sharedStatement  */
  symbol_variable = 105,                   /* variable  */
  symbol_variableStatementList = 106,      /* variableStatementList  */
  symbol_variableStatement = 107,          /* variableStatement  */
  symbol_edgeConnection = 108,             /* edgeConnection  */
  symbol_edgeConnectionComponentType = 109, /* edgeConnectionComponentType  */
  symbol_edgeConnectionStatementList = 110, /* edgeConnectionStatementList  */
  symbol_edgeConnectionStatement = 111,    /* edgeConnectionStatement  */
  symbol_connection = 112,                 /* connection  */
  symbol_connectionComponentType = 113,    /* connectionComponentType  */
  symbol_connectionStatementList = 114,    /* connectionStatementList  */
  symbol_connectionStatement = 115,        /* connectionStatement  */
  symbol_interfaceToMember = 116,          /* interfaceToMember  */
  symbol_psetToMember = 117,               /* psetToMember  */
  symbol_constant = 118,                   /* constant  */
  symbol_constantStatementList = 119,      /* constantStatementList  */
  symbol_constantStatement = 120,          /* constantStatement  */
  symbol_userFunction = 121,               /* userFunction  */
  symbol_userFunctionCall = 122,           /* userFunctionCall  */
  symbol_predicateFunction = 123,          /* predicateFunction  */
  symbol_identifierList = 124,             /* identifierList  */
  symbol_identifierDotList = 125,          /* identifierDotList  */
  symbol_phaseIdentifier = 126,            /* phaseIdentifier  */
  symbol_phaseIdentifierList = 127,        /* phaseIdentifierList  */
  symbol_edgeInstancePhase = 128,          /* edgeInstancePhase  */
  symbol_variableInstancePhase = 129,      /* variableInstancePhase  */
  symbol_nodeInstancePhase = 130,          /* nodeInstancePhase  */
  symbol_sharedPhase = 131,                /* sharedPhase  */
  symbol_interfacePointer = 132,           /* interfacePointer  */
  symbol_interfacePointerList = 133,       /* interfacePointerList  */
  symbol_instanceMapping = 134,            /* instanceMapping  */
  symbol_sharedMapping = 135,              /* sharedMapping  */
  symbol_predicate = 136,                  /* predicate  */
  symbol_dataTypeList = 137,               /* dataTypeList  */
  symbol_optionalDataType = 138,           /* optionalDataType  */
  symbol_dataType = 139,                   /* dataType  */
  symbol_nameComment = 140,                /* nameComment  */
  symbol_nameCommentList = 141,            /* nameCommentList  */
  symbol_nameCommentArgument = 142,        /* nameCommentArgument  */
  symbol_nameCommentArgumentList = 143,    /* nameCommentArgumentList  */
  symbol_functor = 144,                    /* functor  */
  symbol_functorStatementList = 145,       /* functorStatementList  */
  symbol_functorStatement = 146,           /* functorStatement  */
  symbol_execute = 147,                    /* execute  */
  symbol_returnType = 148,                 /* returnType  */
  symbol_initialize = 149,                 /* initialize  */
  symbol_argumentDataTypeList = 150,       /* argumentDataTypeList  */
  symbol_argumentDataType = 151,           /* argumentDataType  */
  symbol_triggeredFunctionInstance = 152,  /* triggeredFunctionInstance  */
  symbol_triggeredFunctionShared = 153,    /* triggeredFunctionShared  */
  symbol_nonPointerTypeClassifier = 154,   /* nonPointerTypeClassifier  */
  symbol_typeClassifier = 155,             /* typeClassifier  */
  symbol_typeCore = 156,                   /* typeCore  */
  symbol_array = 157                       /* array  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;


/* Second part of user prologue.  */
#line 179 "mdl.y"

#endif

#line 396 "mdl.tab.c"


#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if 1

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
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
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
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* 1 */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined MDLLTYPE_IS_TRIVIAL && MDLLTYPE_IS_TRIVIAL \
             && defined MDLSTYPE_IS_TRIVIAL && MDLSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE) \
             + YYSIZEOF (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

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
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  511

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   332


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : symbol_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
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

#if MDLDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   339,   339,   344,   347,   352,   355,   358,   361,   364,
     367,   370,   374,   391,   396,   403,   410,   415,   422,   426,
     432,   436,   440,   444,   448,   452,   456,   460,   464,   468,
     472,   476,   480,   486,   491,   498,   502,   508,   512,   516,
     520,   524,   528,   532,   536,   540,   544,   548,   552,   556,
     562,   568,   574,   580,   584,   591,   595,   601,   605,   609,
     613,   617,   623,   628,   635,   639,   645,   649,   653,   657,
     661,   665,   669,   673,   677,   681,   685,   691,   695,   699,
     705,   713,   716,   722,   726,   732,   736,   742,   748,   756,
     759,   762,   765,   770,   774,   780,   784,   788,   792,   798,
     804,   812,   817,   824,   829,   836,   840,   846,   850,   854,
     858,   864,   870,   877,   883,   888,   895,   900,   907,   912,
     917,   924,   928,   934,   939,   944,   949,   956,   961,   966,
     971,   978,   983,   988,   993,   998,  1005,  1010,  1015,  1020,
    1027,  1034,  1038,  1044,  1050,  1058,  1064,  1073,  1076,  1080,
    1084,  1088,  1092,  1098,  1103,  1108,  1111,  1114,  1117,  1120,
    1123,  1126,  1129,  1132,  1135,  1140,  1144,  1150,  1154,  1160,
    1164,  1170,  1175,  1180,  1185,  1191,  1197,  1205,  1209,  1215,
    1222,  1228,  1233,  1239,  1245,  1254,  1258,  1264,  1268,  1272,
    1278,  1282,  1286,  1290,  1296,  1300,  1306,  1310,  1314,  1318,
    1324,  1328,  1334,  1340,  1344,  1348,  1354,  1358,  1362,  1373,
    1377,  1383,  1387,  1391,  1395,  1401,  1405,  1409,  1413,  1417,
    1421,  1425,  1429,  1433,  1437,  1441,  1445,  1449,  1453,  1457,
    1461,  1465,  1469,  1473,  1477,  1481,  1485,  1489,  1493,  1500
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if 1
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "DOUBLE_CONSTANT",
  "INT_CONSTANT", "STRING_LITERAL", "IDENTIFIER", "STRING", "BOOL", "CHAR",
  "SHORT", "INT", "LONG", "FLOAT", "DOUBLE", "UNSIGNED", "EDGE", "EDGESET",
  "EDGETYPE", "FUNCTOR", "GRID", "NODE", "NODESET", "NODETYPE", "SERVICE",
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
  "nonPointerTypeClassifier", "typeClassifier", "typeCore", "array", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-404)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-214)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
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

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
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

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
       0,    10,    11,    12,    13,    14,    15,    87,    88,    16,
     122,   123,   109,    90,    91,    92,   268,   189,    17,   156,
     157,    93,   259,   464,   465,   128,   299,   470,   471,   472,
     473,    18,   144,   145,    94,   474,    95,   174,   364,   197,
     198,    96,   161,   131,   190,    46,    47,    97,    98,   376,
     137,    99,   138,   213,   214,   349,   350,    19,   110,   111,
     112,   113,   114,   290,   291,   101,   193,   205,   102,   103,
     104
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
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

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
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

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
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

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
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


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = MDLEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == MDLEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (&yylloc, parm, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use MDLerror or MDLUNDEF. */
#define YYERRCODE MDLUNDEF

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if MDLDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YYLOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

# ifndef YYLOCATION_PRINT

#  if defined YY_LOCATION_PRINT

   /* Temporary convenience wrapper in case some people defined the
      undocumented and private YY_LOCATION_PRINT macros.  */
#   define YYLOCATION_PRINT(File, Loc)  YY_LOCATION_PRINT(File, *(Loc))

#  elif defined MDLLTYPE_IS_TRIVIAL && MDLLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static int
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  int res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
}

#   define YYLOCATION_PRINT  yy_location_print_

    /* Temporary convenience wrapper in case some people defined the
       undocumented and private YY_LOCATION_PRINT macros.  */
#   define YY_LOCATION_PRINT(File, Loc)  YYLOCATION_PRINT(File, &(Loc))

#  else

#   define YYLOCATION_PRINT(File, Loc) ((void) 0)
    /* Temporary convenience wrapper in case some people defined the
       undocumented and private YY_LOCATION_PRINT macros.  */
#   define YY_LOCATION_PRINT  YYLOCATION_PRINT

#  endif
# endif /* !defined YYLOCATION_PRINT */


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, Location, parm); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, void* parm)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  YY_USE (yylocationp);
  YY_USE (parm);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, void* parm)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  YYLOCATION_PRINT (yyo, yylocationp);
  YYFPRINTF (yyo, ": ");
  yy_symbol_value_print (yyo, yykind, yyvaluep, yylocationp, parm);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp,
                 int yyrule, void* parm)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)],
                       &(yylsp[(yyi + 1) - (yynrhs)]), parm);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule, parm); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !MDLDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !MDLDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
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


/* Context of a parse error.  */
typedef struct
{
  yy_state_t *yyssp;
  yysymbol_kind_t yytoken;
  YYLTYPE *yylloc;
} yypcontext_t;

/* Put in YYARG at most YYARGN of the expected tokens given the
   current YYCTX, and return the number of tokens stored in YYARG.  If
   YYARG is null, return the number of expected tokens (guaranteed to
   be less than YYNTOKENS).  Return YYENOMEM on memory exhaustion.
   Return 0 if there are more than YYARGN expected tokens, yet fill
   YYARG up to YYARGN. */
static int
yypcontext_expected_tokens (const yypcontext_t *yyctx,
                            yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  int yyn = yypact[+*yyctx->yyssp];
  if (!yypact_value_is_default (yyn))
    {
      /* Start YYX at -YYN if negative to avoid negative indexes in
         YYCHECK.  In other words, skip the first -YYN actions for
         this state because they are default actions.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;
      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yyx;
      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
        if (yycheck[yyx + yyn] == yyx && yyx != symbol_YYerror
            && !yytable_value_is_error (yytable[yyx + yyn]))
          {
            if (!yyarg)
              ++yycount;
            else if (yycount == yyargn)
              return 0;
            else
              yyarg[yycount++] = YY_CAST (yysymbol_kind_t, yyx);
          }
    }
  if (yyarg && yycount == 0 && 0 < yyargn)
    yyarg[0] = symbol_YYEMPTY;
  return yycount;
}




#ifndef yystrlen
# if defined __GLIBC__ && defined _STRING_H
#  define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
# else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
# endif
#endif

#ifndef yystpcpy
# if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#  define yystpcpy stpcpy
# else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
# endif
#endif

#ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYPTRDIFF_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYPTRDIFF_T yyn = 0;
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
            else
              goto append;

          append:
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

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return yystrlen (yystr);
}
#endif


static int
yy_syntax_error_arguments (const yypcontext_t *yyctx,
                           yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yyctx->yytoken != symbol_YYEMPTY)
    {
      int yyn;
      if (yyarg)
        yyarg[yycount] = yyctx->yytoken;
      ++yycount;
      yyn = yypcontext_expected_tokens (yyctx,
                                        yyarg ? yyarg + 1 : yyarg, yyargn - 1);
      if (yyn == YYENOMEM)
        return YYENOMEM;
      else
        yycount += yyn;
    }
  return yycount;
}

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return -1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return YYENOMEM if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                const yypcontext_t *yyctx)
{
  enum { YYARGS_MAX = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  yysymbol_kind_t yyarg[YYARGS_MAX];
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

  /* Actual size of YYARG. */
  int yycount = yy_syntax_error_arguments (yyctx, yyarg, YYARGS_MAX);
  if (yycount == YYENOMEM)
    return YYENOMEM;

  switch (yycount)
    {
#define YYCASE_(N, S)                       \
      case N:                               \
        yyformat = S;                       \
        break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
    }

  /* Compute error message size.  Don't count the "%s"s, but reserve
     room for the terminator.  */
  yysize = yystrlen (yyformat) - 2 * yycount + 1;
  {
    int yyi;
    for (yyi = 0; yyi < yycount; ++yyi)
      {
        YYPTRDIFF_T yysize1
          = yysize + yytnamerr (YY_NULLPTR, yytname[yyarg[yyi]]);
        if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
          yysize = yysize1;
        else
          return YYENOMEM;
      }
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return -1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yytname[yyarg[yyi++]]);
          yyformat += 2;
        }
      else
        {
          ++yyp;
          ++yyformat;
        }
  }
  return 0;
}


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, YYLTYPE *yylocationp, void* parm)
{
  YY_USE (yyvaluep);
  YY_USE (yylocationp);
  YY_USE (parm);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}






/*----------.
| yyparse.  |
`----------*/

int
yyparse (void* parm)
{
/* Lookahead token kind.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

/* Location data for the lookahead symbol.  */
static YYLTYPE yyloc_default
# if defined MDLLTYPE_IS_TRIVIAL && MDLLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
YYLTYPE yylloc = yyloc_default;

    /* Number of syntax errors so far.  */
    int yynerrs = 0;

    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

    /* The location stack: array, bottom, top.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls = yylsa;
    YYLTYPE *yylsp = yyls;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = symbol_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

  /* The locations where the error started and ended.  */
  YYLTYPE yyerror_range[3];

  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = MDLEMPTY; /* Cause a token to be read.  */

  yylsp[0] = yylloc;
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yyls1, yysize * YYSIZEOF (*yylsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
        yyls = yyls1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == MDLEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex (&yylval, &yylloc, parm);
    }

  if (yychar <= MDLEOF)
    {
      yychar = MDLEOF;
      yytoken = symbol_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == MDLerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = MDLUNDEF;
      yytoken = symbol_YYerror;
      yyerror_range[1] = yylloc;
      goto yyerrlab1;
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
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;

  /* Discard the shifted token.  */
  yychar = MDLEMPTY;
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
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location. */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  yyerror_range[1] = yyloc;
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* mdlFile: parserLineList  */
#line 339 "mdl.y"
                         {

}
#line 2471 "mdl.tab.c"
    break;

  case 3: /* parserLineList: parserLine  */
#line 344 "mdl.y"
                           {

}
#line 2479 "mdl.tab.c"
    break;

  case 4: /* parserLineList: parserLineList parserLine  */
#line 347 "mdl.y"
                            {

}
#line 2487 "mdl.tab.c"
    break;

  case 5: /* parserLine: struct  */
#line 352 "mdl.y"
                   {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[0].P_struct));
}
#line 2495 "mdl.tab.c"
    break;

  case 6: /* parserLine: interface  */
#line 355 "mdl.y"
            {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[0].P_interface));
}
#line 2503 "mdl.tab.c"
    break;

  case 7: /* parserLine: edge  */
#line 358 "mdl.y"
       {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[0].P_edge));
}
#line 2511 "mdl.tab.c"
    break;

  case 8: /* parserLine: node  */
#line 361 "mdl.y"
       {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[0].P_node));
}
#line 2519 "mdl.tab.c"
    break;

  case 9: /* parserLine: variable  */
#line 364 "mdl.y"
           {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[0].P_variable));
}
#line 2527 "mdl.tab.c"
    break;

  case 10: /* parserLine: constant  */
#line 367 "mdl.y"
           {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[0].P_constant));
}
#line 2535 "mdl.tab.c"
    break;

  case 11: /* parserLine: functor  */
#line 370 "mdl.y"
          {
   HIGH_LEVEL_EXECUTE(parm, (yyvsp[0].P_functor));
}
#line 2543 "mdl.tab.c"
    break;

  case 12: /* parserLine: error  */
#line 374 "mdl.y"
        {
   MdlContext *c = (MdlContext *) parm;
   MdlLexer *l = c->_lexer;
   
   // Update the error location first
   c->setLastError(l->currentFileName, l->lineCount);
   
   if (!c->isErrorDisplayed()){
      cerr<< "ERROR: ["<<c->getLastErrorFileName()<<":" <<c->getLastErrorLine()<<"] ";
      cerr<< "Unexpected token: " << l->getToken() << endl << endl;
      c->setErrorDisplayed(true);
   }
   c->setError();
}
#line 2562 "mdl.tab.c"
    break;

  case 13: /* struct: STRUCT IDENTIFIER '{' dataTypeList '}'  */
#line 391 "mdl.y"
                                                {
   (yyval.P_struct) = new C_struct(*(yyvsp[-3].P_string), (yyvsp[-1].P_dataTypeList));
   (yyval.P_struct)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 2572 "mdl.tab.c"
    break;

  case 14: /* struct: FRAMEWORK STRUCT IDENTIFIER '{' dataTypeList '}'  */
#line 396 "mdl.y"
                                                    {
   (yyval.P_struct) = new C_struct(*(yyvsp[-3].P_string), (yyvsp[-1].P_dataTypeList), true);
   (yyval.P_struct)->setTokenLocation(CURRENTFILE, (yylsp[-5]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 2582 "mdl.tab.c"
    break;

  case 15: /* interface: INTERFACE IDENTIFIER '{' dataTypeList '}'  */
#line 403 "mdl.y"
                                                      {
   (yyval.P_interface) = new C_interface(*(yyvsp[-3].P_string), (yyvsp[-1].P_dataTypeList));
   (yyval.P_interface)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 2592 "mdl.tab.c"
    break;

  case 16: /* edge: EDGE IDENTIFIER IMPLEMENTS interfacePointerList '{' edgeStatementList '}'  */
#line 410 "mdl.y"
                                                                                {
   (yyval.P_edge) = new C_edge(*(yyvsp[-5].P_string), (yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList));
   (yyval.P_edge)->setTokenLocation(CURRENTFILE, (yylsp[-6]).first_line);
   delete (yyvsp[-5].P_string);
}
#line 2602 "mdl.tab.c"
    break;

  case 17: /* edge: EDGE IDENTIFIER '{' edgeStatementList '}'  */
#line 415 "mdl.y"
                                            {
   (yyval.P_edge) = new C_edge(*(yyvsp[-3].P_string), 0, (yyvsp[-1].P_generalList));
   (yyval.P_edge)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 2612 "mdl.tab.c"
    break;

  case 18: /* edgeStatementList: edgeStatement  */
#line 422 "mdl.y"
                                 {
   (yyval.P_generalList) = new C_generalList((yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2621 "mdl.tab.c"
    break;

  case 19: /* edgeStatementList: edgeStatementList edgeStatement  */
#line 426 "mdl.y"
                                  {
   (yyval.P_generalList) = new C_generalList((yyvsp[-1].P_generalList), (yyvsp[0].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 2630 "mdl.tab.c"
    break;

  case 20: /* edgeStatement: noop  */
#line 432 "mdl.y"
                    {
   (yyval.P_general) = (yyvsp[0].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2639 "mdl.tab.c"
    break;

  case 21: /* edgeStatement: dataType  */
#line 436 "mdl.y"
           {
   (yyval.P_general) = (yyvsp[0].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2648 "mdl.tab.c"
    break;

  case 22: /* edgeStatement: optionalDataType  */
#line 440 "mdl.y"
                   {
   (yyval.P_general) = (yyvsp[0].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2657 "mdl.tab.c"
    break;

  case 23: /* edgeStatement: edgeInstancePhase  */
#line 444 "mdl.y"
                    {
   (yyval.P_general) = (yyvsp[0].P_phase);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2666 "mdl.tab.c"
    break;

  case 24: /* edgeStatement: instanceMapping  */
#line 448 "mdl.y"
                  {
   (yyval.P_general) = (yyvsp[0].P_instanceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2675 "mdl.tab.c"
    break;

  case 25: /* edgeStatement: sharedMapping  */
#line 452 "mdl.y"
                {
   (yyval.P_general) = (yyvsp[0].P_sharedMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2684 "mdl.tab.c"
    break;

  case 26: /* edgeStatement: edgeConnection  */
#line 456 "mdl.y"
                 {
   (yyval.P_general) = (yyvsp[0].P_connection);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2693 "mdl.tab.c"
    break;

  case 27: /* edgeStatement: shared  */
#line 460 "mdl.y"
         {
   (yyval.P_general) = (yyvsp[0].P_shared);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2702 "mdl.tab.c"
    break;

  case 28: /* edgeStatement: inAttrPSet  */
#line 464 "mdl.y"
             {
   (yyval.P_general) = (yyvsp[0].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2711 "mdl.tab.c"
    break;

  case 29: /* edgeStatement: outAttrPSet  */
#line 468 "mdl.y"
              {
   (yyval.P_general) = (yyvsp[0].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2720 "mdl.tab.c"
    break;

  case 30: /* edgeStatement: userFunction  */
#line 472 "mdl.y"
               {
   (yyval.P_general) = (yyvsp[0].P_userFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2729 "mdl.tab.c"
    break;

  case 31: /* edgeStatement: predicateFunction  */
#line 476 "mdl.y"
                    {
   (yyval.P_general) = (yyvsp[0].P_predicateFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2738 "mdl.tab.c"
    break;

  case 32: /* edgeStatement: triggeredFunctionInstance  */
#line 480 "mdl.y"
                            {
   (yyval.P_general) = (yyvsp[0].P_triggeredFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2747 "mdl.tab.c"
    break;

  case 33: /* node: NODE IDENTIFIER IMPLEMENTS interfacePointerList '{' nodeStatementList '}'  */
#line 486 "mdl.y"
                                                                                {
   (yyval.P_node) = new C_node(*(yyvsp[-5].P_string), (yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList));
   (yyval.P_node)->setTokenLocation(CURRENTFILE, (yylsp[-6]).first_line);
   delete (yyvsp[-5].P_string);
}
#line 2757 "mdl.tab.c"
    break;

  case 34: /* node: NODE IDENTIFIER '{' nodeStatementList '}'  */
#line 491 "mdl.y"
                                            {
   (yyval.P_node) = new C_node(*(yyvsp[-3].P_string), 0, (yyvsp[-1].P_generalList));
   (yyval.P_node)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 2767 "mdl.tab.c"
    break;

  case 35: /* nodeStatementList: nodeStatement  */
#line 498 "mdl.y"
                                 {
   (yyval.P_generalList) = new C_generalList((yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2776 "mdl.tab.c"
    break;

  case 36: /* nodeStatementList: nodeStatementList nodeStatement  */
#line 502 "mdl.y"
                                  {
   (yyval.P_generalList) = new C_generalList((yyvsp[-1].P_generalList), (yyvsp[0].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 2785 "mdl.tab.c"
    break;

  case 37: /* nodeStatement: noop  */
#line 508 "mdl.y"
                    {
   (yyval.P_general) = (yyvsp[0].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2794 "mdl.tab.c"
    break;

  case 38: /* nodeStatement: dataType  */
#line 512 "mdl.y"
           {
   (yyval.P_general) = (yyvsp[0].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2803 "mdl.tab.c"
    break;

  case 39: /* nodeStatement: optionalDataType  */
#line 516 "mdl.y"
                   {
   (yyval.P_general) = (yyvsp[0].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2812 "mdl.tab.c"
    break;

  case 40: /* nodeStatement: nodeInstancePhase  */
#line 520 "mdl.y"
                    {
   (yyval.P_general) = (yyvsp[0].P_phase);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2821 "mdl.tab.c"
    break;

  case 41: /* nodeStatement: instanceMapping  */
#line 524 "mdl.y"
                  {
   (yyval.P_general) = (yyvsp[0].P_instanceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2830 "mdl.tab.c"
    break;

  case 42: /* nodeStatement: sharedMapping  */
#line 528 "mdl.y"
                {
   (yyval.P_general) = (yyvsp[0].P_sharedMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2839 "mdl.tab.c"
    break;

  case 43: /* nodeStatement: connection  */
#line 532 "mdl.y"
             {
   (yyval.P_general) = (yyvsp[0].P_connection);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2848 "mdl.tab.c"
    break;

  case 44: /* nodeStatement: shared  */
#line 536 "mdl.y"
         {
   (yyval.P_general) = (yyvsp[0].P_shared);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2857 "mdl.tab.c"
    break;

  case 45: /* nodeStatement: inAttrPSet  */
#line 540 "mdl.y"
             {
   (yyval.P_general) = (yyvsp[0].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2866 "mdl.tab.c"
    break;

  case 46: /* nodeStatement: outAttrPSet  */
#line 544 "mdl.y"
              {
   (yyval.P_general) = (yyvsp[0].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2875 "mdl.tab.c"
    break;

  case 47: /* nodeStatement: userFunction  */
#line 548 "mdl.y"
               {
   (yyval.P_general) = (yyvsp[0].P_userFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2884 "mdl.tab.c"
    break;

  case 48: /* nodeStatement: predicateFunction  */
#line 552 "mdl.y"
                    {
   (yyval.P_general) = (yyvsp[0].P_predicateFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2893 "mdl.tab.c"
    break;

  case 49: /* nodeStatement: triggeredFunctionInstance  */
#line 556 "mdl.y"
                            {
   (yyval.P_general) = (yyvsp[0].P_triggeredFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2902 "mdl.tab.c"
    break;

  case 50: /* noop: ';'  */
#line 562 "mdl.y"
          {
   (yyval.P_noop) = new C_noop();
   (yyval.P_noop)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2911 "mdl.tab.c"
    break;

  case 51: /* inAttrPSet: INATTRPSET '{' dataTypeList '}'  */
#line 568 "mdl.y"
                                             {
   (yyval.P_general) = new C_inAttrPSet((yyvsp[-1].P_dataTypeList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 2920 "mdl.tab.c"
    break;

  case 52: /* outAttrPSet: OUTATTRPSET '{' dataTypeList '}'  */
#line 574 "mdl.y"
                                               {
   (yyval.P_general) = new C_outAttrPSet((yyvsp[-1].P_dataTypeList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 2929 "mdl.tab.c"
    break;

  case 53: /* shared: SHARED '{' sharedStatementList '}'  */
#line 580 "mdl.y"
                                           {
   (yyval.P_shared) = new C_shared((yyvsp[-1].P_generalList));
   (yyval.P_shared)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 2938 "mdl.tab.c"
    break;

  case 54: /* shared: SHARED sharedStatement  */
#line 584 "mdl.y"
                         {
   (yyval.P_shared) = new C_shared();
   (yyval.P_shared)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
   (yyval.P_shared)->setGeneral((yyvsp[0].P_general));
}
#line 2948 "mdl.tab.c"
    break;

  case 55: /* sharedStatementList: sharedStatement  */
#line 591 "mdl.y"
                                     {
   (yyval.P_generalList) = new C_generalList((yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2957 "mdl.tab.c"
    break;

  case 56: /* sharedStatementList: sharedStatementList sharedStatement  */
#line 595 "mdl.y"
                                      {
   (yyval.P_generalList) = new C_generalList((yyvsp[-1].P_generalList), (yyvsp[0].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 2966 "mdl.tab.c"
    break;

  case 57: /* sharedStatement: noop  */
#line 601 "mdl.y"
                      {
   (yyval.P_general) = (yyvsp[0].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2975 "mdl.tab.c"
    break;

  case 58: /* sharedStatement: dataType  */
#line 605 "mdl.y"
           {
   (yyval.P_general) = (yyvsp[0].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2984 "mdl.tab.c"
    break;

  case 59: /* sharedStatement: sharedPhase  */
#line 609 "mdl.y"
              {
   (yyval.P_general) = (yyvsp[0].P_phase);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 2993 "mdl.tab.c"
    break;

  case 60: /* sharedStatement: triggeredFunctionShared  */
#line 613 "mdl.y"
                          {
   (yyval.P_general) = (yyvsp[0].P_triggeredFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3002 "mdl.tab.c"
    break;

  case 61: /* sharedStatement: optionalDataType  */
#line 617 "mdl.y"
                   {
   (yyval.P_general) = (yyvsp[0].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3011 "mdl.tab.c"
    break;

  case 62: /* variable: VARIABLE IDENTIFIER IMPLEMENTS interfacePointerList '{' variableStatementList '}'  */
#line 623 "mdl.y"
                                                                                            {
   (yyval.P_variable) = new C_variable(*(yyvsp[-5].P_string), (yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList));
   (yyval.P_variable)->setTokenLocation(CURRENTFILE, (yylsp[-6]).first_line);
   delete (yyvsp[-5].P_string);
}
#line 3021 "mdl.tab.c"
    break;

  case 63: /* variable: VARIABLE IDENTIFIER '{' variableStatementList '}'  */
#line 628 "mdl.y"
                                                    {
   (yyval.P_variable) = new C_variable(*(yyvsp[-3].P_string), 0, (yyvsp[-1].P_generalList));
   (yyval.P_variable)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 3031 "mdl.tab.c"
    break;

  case 64: /* variableStatementList: variableStatement  */
#line 635 "mdl.y"
                                         {
   (yyval.P_generalList) = new C_generalList((yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3040 "mdl.tab.c"
    break;

  case 65: /* variableStatementList: variableStatementList variableStatement  */
#line 639 "mdl.y"
                                          {
   (yyval.P_generalList) = new C_generalList((yyvsp[-1].P_generalList), (yyvsp[0].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 3049 "mdl.tab.c"
    break;

  case 66: /* variableStatement: noop  */
#line 645 "mdl.y"
                        {
   (yyval.P_general) = (yyvsp[0].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3058 "mdl.tab.c"
    break;

  case 67: /* variableStatement: dataType  */
#line 649 "mdl.y"
           {
   (yyval.P_general) = (yyvsp[0].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3067 "mdl.tab.c"
    break;

  case 68: /* variableStatement: optionalDataType  */
#line 653 "mdl.y"
                   {
   (yyval.P_general) = (yyvsp[0].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3076 "mdl.tab.c"
    break;

  case 69: /* variableStatement: variableInstancePhase  */
#line 657 "mdl.y"
                        {
   (yyval.P_general) = (yyvsp[0].P_phase);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3085 "mdl.tab.c"
    break;

  case 70: /* variableStatement: instanceMapping  */
#line 661 "mdl.y"
                  {
   (yyval.P_general) = (yyvsp[0].P_instanceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3094 "mdl.tab.c"
    break;

  case 71: /* variableStatement: connection  */
#line 665 "mdl.y"
             {
   (yyval.P_general) = (yyvsp[0].P_connection);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3103 "mdl.tab.c"
    break;

  case 72: /* variableStatement: inAttrPSet  */
#line 669 "mdl.y"
             {
   (yyval.P_general) = (yyvsp[0].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3112 "mdl.tab.c"
    break;

  case 73: /* variableStatement: outAttrPSet  */
#line 673 "mdl.y"
              {
   (yyval.P_general) = (yyvsp[0].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3121 "mdl.tab.c"
    break;

  case 74: /* variableStatement: userFunction  */
#line 677 "mdl.y"
               {
   (yyval.P_general) = (yyvsp[0].P_userFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3130 "mdl.tab.c"
    break;

  case 75: /* variableStatement: predicateFunction  */
#line 681 "mdl.y"
                    {
   (yyval.P_general) = (yyvsp[0].P_predicateFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3139 "mdl.tab.c"
    break;

  case 76: /* variableStatement: triggeredFunctionInstance  */
#line 685 "mdl.y"
                            {
   (yyval.P_general) = (yyvsp[0].P_triggeredFunction);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3148 "mdl.tab.c"
    break;

  case 77: /* edgeConnection: CONNECTION PRE NODE '(' ')' EXPECTS interfacePointerList '{' edgeConnectionStatementList '}'  */
#line 691 "mdl.y"
                                                                                                             {
   (yyval.P_connection) = new C_edgeConnection((yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList), Connection::_PRE);
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[-9]).first_line);
}
#line 3157 "mdl.tab.c"
    break;

  case 78: /* edgeConnection: CONNECTION POST NODE '(' ')' EXPECTS interfacePointerList '{' edgeConnectionStatementList '}'  */
#line 695 "mdl.y"
                                                                                                {
   (yyval.P_connection) = new C_edgeConnection((yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList), Connection::_POST);
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[-9]).first_line);
}
#line 3166 "mdl.tab.c"
    break;

  case 79: /* edgeConnection: CONNECTION PRE edgeConnectionComponentType '(' ')' EXPECTS interfacePointerList '{' connectionStatementList '}'  */
#line 699 "mdl.y"
                                                                                                                  {
   (yyval.P_connection) = new C_regularConnection((yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList),
				(yyvsp[-7].V_connectionComponentType),
				Connection::_PRE);
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[-9]).first_line);
}
#line 3177 "mdl.tab.c"
    break;

  case 80: /* edgeConnection: CONNECTION PRE edgeConnectionComponentType '(' predicate ')' EXPECTS interfacePointerList '{' connectionStatementList '}'  */
#line 705 "mdl.y"
                                                                                                                            {
   (yyval.P_connection) = new C_regularConnection((yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList),
				(yyvsp[-8].V_connectionComponentType),
				Connection::_PRE, (yyvsp[-6].P_predicate));
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[-10]).first_line);
}
#line 3188 "mdl.tab.c"
    break;

  case 81: /* edgeConnectionComponentType: CONSTANT  */
#line 713 "mdl.y"
                                      {
   (yyval.V_connectionComponentType) = Connection::_CONSTANT;
}
#line 3196 "mdl.tab.c"
    break;

  case 82: /* edgeConnectionComponentType: VARIABLE  */
#line 716 "mdl.y"
           {
   (yyval.V_connectionComponentType) = Connection::_VARIABLE;
}
#line 3204 "mdl.tab.c"
    break;

  case 83: /* edgeConnectionStatementList: edgeConnectionStatement  */
#line 722 "mdl.y"
                                                     {
   (yyval.P_generalList) = new C_generalList((yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3213 "mdl.tab.c"
    break;

  case 84: /* edgeConnectionStatementList: edgeConnectionStatementList edgeConnectionStatement  */
#line 726 "mdl.y"
                                                      {
   (yyval.P_generalList) = new C_generalList((yyvsp[-1].P_generalList), (yyvsp[0].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 3222 "mdl.tab.c"
    break;

  case 85: /* edgeConnectionStatement: noop  */
#line 732 "mdl.y"
                              {
   (yyval.P_general) = (yyvsp[0].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3231 "mdl.tab.c"
    break;

  case 86: /* edgeConnectionStatement: interfaceToMember  */
#line 736 "mdl.y"
                    {
   (yyval.P_general) = (yyvsp[0].P_interfaceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3240 "mdl.tab.c"
    break;

  case 87: /* connection: CONNECTION PRE connectionComponentType '(' ')' EXPECTS interfacePointerList '{' connectionStatementList '}'  */
#line 742 "mdl.y"
                                                                                                                        {
   (yyval.P_connection) = new C_regularConnection((yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList),
				(yyvsp[-7].V_connectionComponentType),
				Connection::_PRE);
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[-9]).first_line);
}
#line 3251 "mdl.tab.c"
    break;

  case 88: /* connection: CONNECTION PRE connectionComponentType '(' predicate ')' EXPECTS interfacePointerList '{' connectionStatementList '}'  */
#line 748 "mdl.y"
                                                                                                                        {
   (yyval.P_connection) = new C_regularConnection((yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList),
				(yyvsp[-8].V_connectionComponentType),
				Connection::_PRE, (yyvsp[-6].P_predicate));
   (yyval.P_connection)->setTokenLocation(CURRENTFILE, (yylsp[-10]).first_line);
}
#line 3262 "mdl.tab.c"
    break;

  case 89: /* connectionComponentType: NODE  */
#line 756 "mdl.y"
                              {
   (yyval.V_connectionComponentType) = Connection::_NODE;
}
#line 3270 "mdl.tab.c"
    break;

  case 90: /* connectionComponentType: EDGE  */
#line 759 "mdl.y"
       {
   (yyval.V_connectionComponentType) = Connection::_EDGE;
}
#line 3278 "mdl.tab.c"
    break;

  case 91: /* connectionComponentType: CONSTANT  */
#line 762 "mdl.y"
           {
   (yyval.V_connectionComponentType) = Connection::_CONSTANT;
}
#line 3286 "mdl.tab.c"
    break;

  case 92: /* connectionComponentType: VARIABLE  */
#line 765 "mdl.y"
           {
   (yyval.V_connectionComponentType) = Connection::_VARIABLE;
}
#line 3294 "mdl.tab.c"
    break;

  case 93: /* connectionStatementList: connectionStatement  */
#line 770 "mdl.y"
                                             {
   (yyval.P_generalList) = new C_generalList((yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3303 "mdl.tab.c"
    break;

  case 94: /* connectionStatementList: connectionStatementList connectionStatement  */
#line 774 "mdl.y"
                                              {
   (yyval.P_generalList) = new C_generalList((yyvsp[-1].P_generalList), (yyvsp[0].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 3312 "mdl.tab.c"
    break;

  case 95: /* connectionStatement: noop  */
#line 780 "mdl.y"
                          {
   (yyval.P_general) = (yyvsp[0].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3321 "mdl.tab.c"
    break;

  case 96: /* connectionStatement: interfaceToMember  */
#line 784 "mdl.y"
                    {
   (yyval.P_general) = (yyvsp[0].P_interfaceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3330 "mdl.tab.c"
    break;

  case 97: /* connectionStatement: psetToMember  */
#line 788 "mdl.y"
               {
   (yyval.P_general) = (yyvsp[0].P_psetMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3339 "mdl.tab.c"
    break;

  case 98: /* connectionStatement: userFunctionCall  */
#line 792 "mdl.y"
                   {
   (yyval.P_general) = (yyvsp[0].P_userFunctionCall);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3348 "mdl.tab.c"
    break;

  case 99: /* interfaceToMember: IDENTIFIER DOT IDENTIFIER RIGHTSHIFT identifierDotList ';'  */
#line 798 "mdl.y"
                                                                              {
   (yyval.P_interfaceMapping) = new C_interfaceToInstance(*(yyvsp[-5].P_string), *(yyvsp[-3].P_string), (yyvsp[-1].P_identifierList));
   (yyval.P_interfaceMapping)->setTokenLocation(CURRENTFILE, (yylsp[-5]).first_line);
   delete (yyvsp[-5].P_string);
   delete (yyvsp[-3].P_string);
}
#line 3359 "mdl.tab.c"
    break;

  case 100: /* interfaceToMember: IDENTIFIER DOT IDENTIFIER RIGHTSHIFT SHARED DOT identifierDotList ';'  */
#line 804 "mdl.y"
                                                                        {
   (yyval.P_interfaceMapping) = new C_interfaceToShared(*(yyvsp[-7].P_string), *(yyvsp[-5].P_string), (yyvsp[-1].P_identifierList));
   (yyval.P_interfaceMapping)->setTokenLocation(CURRENTFILE, (yylsp[-7]).first_line);
   delete (yyvsp[-7].P_string);
   delete (yyvsp[-5].P_string);
}
#line 3370 "mdl.tab.c"
    break;

  case 101: /* psetToMember: PSET DOT IDENTIFIER RIGHTSHIFT identifierDotList ';'  */
#line 812 "mdl.y"
                                                                   {
   (yyval.P_psetMapping) = new C_psetToInstance(*(yyvsp[-3].P_string), (yyvsp[-1].P_identifierList));
   (yyval.P_psetMapping)->setTokenLocation(CURRENTFILE, (yylsp[-5]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 3380 "mdl.tab.c"
    break;

  case 102: /* psetToMember: PSET DOT IDENTIFIER RIGHTSHIFT SHARED DOT identifierDotList ';'  */
#line 817 "mdl.y"
                                                                  {
   (yyval.P_psetMapping) = new C_psetToShared(*(yyvsp[-5].P_string), (yyvsp[-1].P_identifierList));
   (yyval.P_psetMapping)->setTokenLocation(CURRENTFILE, (yylsp[-7]).first_line);
   delete (yyvsp[-5].P_string);
}
#line 3390 "mdl.tab.c"
    break;

  case 103: /* constant: CONSTANT IDENTIFIER IMPLEMENTS interfacePointerList '{' constantStatementList '}'  */
#line 824 "mdl.y"
                                                                                            {
   (yyval.P_constant) = new C_constant(*(yyvsp[-5].P_string), (yyvsp[-3].P_interfacePointerList), (yyvsp[-1].P_generalList));
   (yyval.P_constant)->setTokenLocation(CURRENTFILE, (yylsp[-6]).first_line);
   delete (yyvsp[-5].P_string);
}
#line 3400 "mdl.tab.c"
    break;

  case 104: /* constant: CONSTANT IDENTIFIER '{' constantStatementList '}'  */
#line 829 "mdl.y"
                                                    {
   (yyval.P_constant) = new C_constant(*(yyvsp[-3].P_string), 0, (yyvsp[-1].P_generalList));
   (yyval.P_constant)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 3410 "mdl.tab.c"
    break;

  case 105: /* constantStatementList: constantStatement  */
#line 836 "mdl.y"
                                         {
   (yyval.P_generalList) = new C_generalList((yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3419 "mdl.tab.c"
    break;

  case 106: /* constantStatementList: constantStatementList constantStatement  */
#line 840 "mdl.y"
                                          {
   (yyval.P_generalList) = new C_generalList((yyvsp[-1].P_generalList), (yyvsp[0].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 3428 "mdl.tab.c"
    break;

  case 107: /* constantStatement: noop  */
#line 846 "mdl.y"
                        {
   (yyval.P_general) = (yyvsp[0].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3437 "mdl.tab.c"
    break;

  case 108: /* constantStatement: dataType  */
#line 850 "mdl.y"
           {
   (yyval.P_general) = (yyvsp[0].P_dataType);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3446 "mdl.tab.c"
    break;

  case 109: /* constantStatement: instanceMapping  */
#line 854 "mdl.y"
                  {
   (yyval.P_general) = (yyvsp[0].P_instanceMapping);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3455 "mdl.tab.c"
    break;

  case 110: /* constantStatement: outAttrPSet  */
#line 858 "mdl.y"
              {
   (yyval.P_general) = (yyvsp[0].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3464 "mdl.tab.c"
    break;

  case 111: /* userFunction: USERFUNCTION identifierList ';'  */
#line 864 "mdl.y"
                                              {
   (yyval.P_userFunction) = new C_userFunction((yyvsp[-1].P_identifierList));
   (yyval.P_userFunction)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3473 "mdl.tab.c"
    break;

  case 112: /* userFunctionCall: IDENTIFIER '(' ')' ';'  */
#line 870 "mdl.y"
                                         {
   (yyval.P_userFunctionCall) = new C_userFunctionCall(*(yyvsp[-3].P_string));
   (yyval.P_userFunctionCall)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 3483 "mdl.tab.c"
    break;

  case 113: /* predicateFunction: PREDICATEFUNCTION identifierList ';'  */
#line 877 "mdl.y"
                                                        {
   (yyval.P_predicateFunction) = new C_predicateFunction((yyvsp[-1].P_identifierList));
   (yyval.P_predicateFunction)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3492 "mdl.tab.c"
    break;

  case 114: /* identifierList: IDENTIFIER  */
#line 883 "mdl.y"
                           {
   (yyval.P_identifierList) = new C_identifierList(*(yyvsp[0].P_string));
   (yyval.P_identifierList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
   delete (yyvsp[0].P_string);
}
#line 3502 "mdl.tab.c"
    break;

  case 115: /* identifierList: identifierList ',' IDENTIFIER  */
#line 888 "mdl.y"
                                {
   (yyval.P_identifierList) = new C_identifierList((yyvsp[-2].P_identifierList), *(yyvsp[0].P_string));
   (yyval.P_identifierList)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
   delete (yyvsp[0].P_string);
}
#line 3512 "mdl.tab.c"
    break;

  case 116: /* identifierDotList: IDENTIFIER  */
#line 895 "mdl.y"
                              {
   (yyval.P_identifierList) = new C_identifierList(*(yyvsp[0].P_string));
   (yyval.P_identifierList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
   delete (yyvsp[0].P_string);
}
#line 3522 "mdl.tab.c"
    break;

  case 117: /* identifierDotList: identifierDotList DOT IDENTIFIER  */
#line 900 "mdl.y"
                                   {
   (yyval.P_identifierList) = new C_identifierList((yyvsp[-2].P_identifierList), *(yyvsp[0].P_string));
   (yyval.P_identifierList)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
   delete (yyvsp[0].P_string);
}
#line 3532 "mdl.tab.c"
    break;

  case 118: /* phaseIdentifier: IDENTIFIER  */
#line 907 "mdl.y"
                            {
   (yyval.P_phaseIdentifier) = new C_phaseIdentifier(*(yyvsp[0].P_string));
   (yyval.P_phaseIdentifier)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
   delete (yyvsp[0].P_string);
}
#line 3542 "mdl.tab.c"
    break;

  case 119: /* phaseIdentifier: IDENTIFIER '(' ')'  */
#line 912 "mdl.y"
                     {
   (yyval.P_phaseIdentifier) = new C_phaseIdentifier(*(yyvsp[-2].P_string));
   (yyval.P_phaseIdentifier)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
   delete (yyvsp[-2].P_string);
}
#line 3552 "mdl.tab.c"
    break;

  case 120: /* phaseIdentifier: IDENTIFIER '(' identifierList ')'  */
#line 917 "mdl.y"
                                    {
   (yyval.P_phaseIdentifier) = new C_phaseIdentifier(*(yyvsp[-3].P_string), (yyvsp[-1].P_identifierList));
   (yyval.P_phaseIdentifier)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 3562 "mdl.tab.c"
    break;

  case 121: /* phaseIdentifierList: phaseIdentifier  */
#line 924 "mdl.y"
                                     {
   (yyval.P_phaseIdentifierList) = new C_phaseIdentifierList((yyvsp[0].P_phaseIdentifier));
   (yyval.P_phaseIdentifierList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3571 "mdl.tab.c"
    break;

  case 122: /* phaseIdentifierList: phaseIdentifierList ',' phaseIdentifier  */
#line 928 "mdl.y"
                                          {
   (yyval.P_phaseIdentifierList) = new C_phaseIdentifierList((yyvsp[-2].P_phaseIdentifierList), (yyvsp[0].P_phaseIdentifier));
   (yyval.P_phaseIdentifierList)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3580 "mdl.tab.c"
    break;

  case 123: /* edgeInstancePhase: INITPHASE phaseIdentifierList ';'  */
#line 934 "mdl.y"
                                                     {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_initPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3590 "mdl.tab.c"
    break;

  case 124: /* edgeInstancePhase: RUNTIMEPHASE phaseIdentifierList ';'  */
#line 939 "mdl.y"
                                       {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3600 "mdl.tab.c"
    break;

  case 125: /* edgeInstancePhase: FINALPHASE phaseIdentifierList ';'  */
#line 944 "mdl.y"
                                     {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_finalPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3610 "mdl.tab.c"
    break;

  case 126: /* edgeInstancePhase: LOADPHASE phaseIdentifierList ';'  */
#line 949 "mdl.y"
                                    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_loadPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3620 "mdl.tab.c"
    break;

  case 127: /* variableInstancePhase: INITPHASE phaseIdentifierList ';'  */
#line 956 "mdl.y"
                                                         {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_initPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3630 "mdl.tab.c"
    break;

  case 128: /* variableInstancePhase: RUNTIMEPHASE phaseIdentifierList ';'  */
#line 961 "mdl.y"
                                       {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3640 "mdl.tab.c"
    break;

  case 129: /* variableInstancePhase: FINALPHASE phaseIdentifierList ';'  */
#line 966 "mdl.y"
                                     {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_finalPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3650 "mdl.tab.c"
    break;

  case 130: /* variableInstancePhase: LOADPHASE phaseIdentifierList ';'  */
#line 971 "mdl.y"
                                    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_loadPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3660 "mdl.tab.c"
    break;

  case 131: /* nodeInstancePhase: INITPHASE phaseIdentifierList ';'  */
#line 978 "mdl.y"
                                                     {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_initPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3670 "mdl.tab.c"
    break;

  case 132: /* nodeInstancePhase: RUNTIMEPHASE phaseIdentifierList ';'  */
#line 983 "mdl.y"
                                       {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3680 "mdl.tab.c"
    break;

  case 133: /* nodeInstancePhase: RUNTIMEPHASE GRIDLAYERS phaseIdentifierList ';'  */
#line 988 "mdl.y"
                                                  {
   std::unique_ptr<PhaseType> pType(new PhaseTypeGridLayers());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 3690 "mdl.tab.c"
    break;

  case 134: /* nodeInstancePhase: FINALPHASE phaseIdentifierList ';'  */
#line 993 "mdl.y"
                                     {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_finalPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3700 "mdl.tab.c"
    break;

  case 135: /* nodeInstancePhase: LOADPHASE phaseIdentifierList ';'  */
#line 998 "mdl.y"
                                    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeInstance());
   (yyval.P_phase) = new C_loadPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3710 "mdl.tab.c"
    break;

  case 136: /* sharedPhase: INITPHASE phaseIdentifierList ';'  */
#line 1005 "mdl.y"
                                               {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   (yyval.P_phase) = new C_initPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3720 "mdl.tab.c"
    break;

  case 137: /* sharedPhase: RUNTIMEPHASE phaseIdentifierList ';'  */
#line 1010 "mdl.y"
                                       {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   (yyval.P_phase) = new C_runtimePhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3730 "mdl.tab.c"
    break;

  case 138: /* sharedPhase: FINALPHASE phaseIdentifierList ';'  */
#line 1015 "mdl.y"
                                     {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   (yyval.P_phase) = new C_finalPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3740 "mdl.tab.c"
    break;

  case 139: /* sharedPhase: LOADPHASE phaseIdentifierList ';'  */
#line 1020 "mdl.y"
                                    {
   std::unique_ptr<PhaseType> pType(new PhaseTypeShared());
   (yyval.P_phase) = new C_loadPhase((yyvsp[-1].P_phaseIdentifierList), std::move(pType));
   (yyval.P_phase)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3750 "mdl.tab.c"
    break;

  case 140: /* interfacePointer: IDENTIFIER  */
#line 1027 "mdl.y"
                             {
   (yyval.P_interfacePointer) = new C_interfacePointer(*(yyvsp[0].P_string));
   (yyval.P_interfacePointer)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
   delete (yyvsp[0].P_string);
}
#line 3760 "mdl.tab.c"
    break;

  case 141: /* interfacePointerList: interfacePointer  */
#line 1034 "mdl.y"
                                       {
   (yyval.P_interfacePointerList) = new C_interfacePointerList((yyvsp[0].P_interfacePointer));
   (yyval.P_interfacePointerList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3769 "mdl.tab.c"
    break;

  case 142: /* interfacePointerList: interfacePointerList ',' interfacePointer  */
#line 1038 "mdl.y"
                                            {
   (yyval.P_interfacePointerList) = new C_interfacePointerList((yyvsp[-2].P_interfacePointerList),(yyvsp[0].P_interfacePointer));
   (yyval.P_interfacePointerList)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 3778 "mdl.tab.c"
    break;

  case 143: /* instanceMapping: IDENTIFIER DOT IDENTIFIER LEFTSHIFT identifierDotList ';'  */
#line 1044 "mdl.y"
                                                                           {
   (yyval.P_instanceMapping) = new C_instanceMapping(*(yyvsp[-5].P_string), *(yyvsp[-3].P_string), (yyvsp[-1].P_identifierList));
   (yyval.P_instanceMapping)->setTokenLocation(CURRENTFILE, (yylsp[-5]).first_line);
   delete (yyvsp[-5].P_string);
   delete (yyvsp[-3].P_string);
}
#line 3789 "mdl.tab.c"
    break;

  case 144: /* instanceMapping: IDENTIFIER DOT IDENTIFIER LEFTSHIFT AMPERSAND identifierDotList ';'  */
#line 1050 "mdl.y"
                                                                      {
   (yyval.P_instanceMapping) = new C_instanceMapping(*(yyvsp[-6].P_string), *(yyvsp[-4].P_string), (yyvsp[-1].P_identifierList), true);
   (yyval.P_instanceMapping)->setTokenLocation(CURRENTFILE, (yylsp[-6]).first_line);
   delete (yyvsp[-6].P_string);
   delete (yyvsp[-4].P_string);
}
#line 3800 "mdl.tab.c"
    break;

  case 145: /* sharedMapping: IDENTIFIER DOT IDENTIFIER LEFTSHIFT SHARED DOT identifierDotList ';'  */
#line 1058 "mdl.y"
                                                                                    {
   (yyval.P_sharedMapping) = new C_sharedMapping(*(yyvsp[-7].P_string), *(yyvsp[-5].P_string), (yyvsp[-1].P_identifierList));
   (yyval.P_sharedMapping)->setTokenLocation(CURRENTFILE, (yylsp[-7]).first_line);
   delete (yyvsp[-7].P_string);
   delete (yyvsp[-5].P_string);
}
#line 3811 "mdl.tab.c"
    break;

  case 146: /* sharedMapping: IDENTIFIER DOT IDENTIFIER LEFTSHIFT AMPERSAND SHARED DOT identifierDotList ';'  */
#line 1064 "mdl.y"
                                                                                 {
   (yyval.P_sharedMapping) = new C_sharedMapping(*(yyvsp[-8].P_string), *(yyvsp[-6].P_string), (yyvsp[-1].P_identifierList), true);
   (yyval.P_sharedMapping)->setTokenLocation(CURRENTFILE, (yylsp[-8]).first_line);
   delete (yyvsp[-8].P_string);
   delete (yyvsp[-6].P_string);
}
#line 3822 "mdl.tab.c"
    break;

  case 147: /* predicate: '(' predicate ')'  */
#line 1073 "mdl.y"
                             {
   (yyval.P_predicate) = new Predicate(new ParanthesisOp(), (yyvsp[-1].P_predicate));  
}
#line 3830 "mdl.tab.c"
    break;

  case 148: /* predicate: IDENTIFIER  */
#line 1076 "mdl.y"
             {
   (yyval.P_predicate) = new InstancePredicate(new TerminalOp(), *(yyvsp[0].P_string));  
   delete (yyvsp[0].P_string);
}
#line 3839 "mdl.tab.c"
    break;

  case 149: /* predicate: IDENTIFIER '(' ')'  */
#line 1080 "mdl.y"
                     {
   (yyval.P_predicate) = new FunctionPredicate(new TerminalOp(), *(yyvsp[-2].P_string));  
   delete (yyvsp[-2].P_string);
}
#line 3848 "mdl.tab.c"
    break;

  case 150: /* predicate: SHARED DOT IDENTIFIER  */
#line 1084 "mdl.y"
                        {
   (yyval.P_predicate) = new SharedPredicate(new TerminalOp(), *(yyvsp[0].P_string));  
   delete (yyvsp[0].P_string);
}
#line 3857 "mdl.tab.c"
    break;

  case 151: /* predicate: PSET DOT IDENTIFIER  */
#line 1088 "mdl.y"
                      {
   (yyval.P_predicate) = new PSetPredicate(new TerminalOp(), *(yyvsp[0].P_string));  
   delete (yyvsp[0].P_string);
}
#line 3866 "mdl.tab.c"
    break;

  case 152: /* predicate: STRING_LITERAL  */
#line 1092 "mdl.y"
                 {
   std::ostringstream os;
   os << '"' << *(yyvsp[0].P_string) << '"';
   (yyval.P_predicate) = new Predicate(new TerminalOp(), os.str(), "string");  
   delete (yyvsp[0].P_string);
}
#line 3877 "mdl.tab.c"
    break;

  case 153: /* predicate: INT_CONSTANT  */
#line 1098 "mdl.y"
               {
   std::ostringstream os;
   os << (yyvsp[0].V_int);
   (yyval.P_predicate) = new Predicate(new TerminalOp(), os.str(), "int");  
}
#line 3887 "mdl.tab.c"
    break;

  case 154: /* predicate: DOUBLE_CONSTANT  */
#line 1103 "mdl.y"
                  {
   std::ostringstream os;
   os << (yyvsp[0].V_double);
   (yyval.P_predicate) = new Predicate(new TerminalOp(), os.str(), "double");  
}
#line 3897 "mdl.tab.c"
    break;

  case 155: /* predicate: _TRUE  */
#line 1108 "mdl.y"
        {
   (yyval.P_predicate) = new Predicate(new TerminalOp(), "true", "bool");  
}
#line 3905 "mdl.tab.c"
    break;

  case 156: /* predicate: _FALSE  */
#line 1111 "mdl.y"
         {
   (yyval.P_predicate) = new Predicate(new TerminalOp(), "false", "bool");  
}
#line 3913 "mdl.tab.c"
    break;

  case 157: /* predicate: predicate EQUAL predicate  */
#line 1114 "mdl.y"
                            {
   (yyval.P_predicate) = new Predicate(new EqualOp(), (yyvsp[-2].P_predicate), (yyvsp[0].P_predicate));  
}
#line 3921 "mdl.tab.c"
    break;

  case 158: /* predicate: predicate NOT_EQUAL predicate  */
#line 1117 "mdl.y"
                                {
   (yyval.P_predicate) = new Predicate(new NotEqualOp(), (yyvsp[-2].P_predicate), (yyvsp[0].P_predicate));  
}
#line 3929 "mdl.tab.c"
    break;

  case 159: /* predicate: predicate LESS_EQUAL predicate  */
#line 1120 "mdl.y"
                                 {
   (yyval.P_predicate) = new Predicate(new LessEqualOp(), (yyvsp[-2].P_predicate), (yyvsp[0].P_predicate));  
}
#line 3937 "mdl.tab.c"
    break;

  case 160: /* predicate: predicate GREATER_EQUAL predicate  */
#line 1123 "mdl.y"
                                    {
   (yyval.P_predicate) = new Predicate(new GreaterEqualOp(), (yyvsp[-2].P_predicate), (yyvsp[0].P_predicate));  
}
#line 3945 "mdl.tab.c"
    break;

  case 161: /* predicate: predicate LESS predicate  */
#line 1126 "mdl.y"
                           {
   (yyval.P_predicate) = new Predicate(new LessOp(), (yyvsp[-2].P_predicate), (yyvsp[0].P_predicate));  
}
#line 3953 "mdl.tab.c"
    break;

  case 162: /* predicate: predicate GREATER predicate  */
#line 1129 "mdl.y"
                              {
   (yyval.P_predicate) = new Predicate(new GreaterOp(), (yyvsp[-2].P_predicate), (yyvsp[0].P_predicate));  
}
#line 3961 "mdl.tab.c"
    break;

  case 163: /* predicate: predicate AND predicate  */
#line 1132 "mdl.y"
                          {
   (yyval.P_predicate) = new Predicate(new AndOp(), (yyvsp[-2].P_predicate), (yyvsp[0].P_predicate));  
}
#line 3969 "mdl.tab.c"
    break;

  case 164: /* predicate: predicate OR predicate  */
#line 1135 "mdl.y"
                         {
   (yyval.P_predicate) = new Predicate(new OrOp(), (yyvsp[-2].P_predicate), (yyvsp[0].P_predicate));  
}
#line 3977 "mdl.tab.c"
    break;

  case 165: /* dataTypeList: dataType  */
#line 1140 "mdl.y"
                       {
   (yyval.P_dataTypeList) = new C_dataTypeList((yyvsp[0].P_dataType));
   (yyval.P_dataTypeList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 3986 "mdl.tab.c"
    break;

  case 166: /* dataTypeList: dataTypeList dataType  */
#line 1144 "mdl.y"
                        {
   (yyval.P_dataTypeList) = new C_dataTypeList((yyvsp[-1].P_dataTypeList), (yyvsp[0].P_dataType));
   (yyval.P_dataTypeList)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 3995 "mdl.tab.c"
    break;

  case 167: /* optionalDataType: OPTIONAL nonPointerTypeClassifier nameCommentList ';'  */
#line 1150 "mdl.y"
                                                                        {
   (yyval.P_dataType) = new C_dataType((yyvsp[-2].P_typeClassifier), (yyvsp[-1].P_nameCommentList), false, true);
   (yyval.P_dataType)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 4004 "mdl.tab.c"
    break;

  case 168: /* optionalDataType: OPTIONAL DERIVED nonPointerTypeClassifier nameCommentList ';'  */
#line 1154 "mdl.y"
                                                                {
   (yyval.P_dataType) = new C_dataType((yyvsp[-2].P_typeClassifier), (yyvsp[-1].P_nameCommentList), true, true);
   (yyval.P_dataType)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);
}
#line 4013 "mdl.tab.c"
    break;

  case 169: /* dataType: typeClassifier nameCommentList ';'  */
#line 1160 "mdl.y"
                                             {
   (yyval.P_dataType) = new C_dataType((yyvsp[-2].P_typeClassifier), (yyvsp[-1].P_nameCommentList));
   (yyval.P_dataType)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 4022 "mdl.tab.c"
    break;

  case 170: /* dataType: DERIVED typeClassifier nameCommentList ';'  */
#line 1164 "mdl.y"
                                             {
   (yyval.P_dataType) = new C_dataType((yyvsp[-2].P_typeClassifier), (yyvsp[-1].P_nameCommentList), true);
   (yyval.P_dataType)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 4031 "mdl.tab.c"
    break;

  case 171: /* nameComment: IDENTIFIER  */
#line 1170 "mdl.y"
                        {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[0].P_string));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
   delete (yyvsp[0].P_string);
}
#line 4041 "mdl.tab.c"
    break;

  case 172: /* nameComment: IDENTIFIER '(' INT_CONSTANT ')'  */
#line 1175 "mdl.y"
                                  {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[-3].P_string), (yyvsp[-1].V_int));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
   delete (yyvsp[-3].P_string);
}
#line 4051 "mdl.tab.c"
    break;

  case 173: /* nameComment: IDENTIFIER '(' INT_CONSTANT ',' INT_CONSTANT ')'  */
#line 1180 "mdl.y"
                                                   {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[-5].P_string), (yyvsp[-3].V_int), (yyvsp[-1].V_int));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[-5]).first_line);
   delete (yyvsp[-5].P_string);
}
#line 4061 "mdl.tab.c"
    break;

  case 174: /* nameComment: IDENTIFIER ':' STRING_LITERAL  */
#line 1185 "mdl.y"
                                {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[-2].P_string), *(yyvsp[0].P_string));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
   delete (yyvsp[-2].P_string);
   delete (yyvsp[0].P_string);
}
#line 4072 "mdl.tab.c"
    break;

  case 175: /* nameComment: IDENTIFIER '(' INT_CONSTANT ')' ':' STRING_LITERAL  */
#line 1191 "mdl.y"
                                                     {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[-5].P_string), *(yyvsp[0].P_string), (yyvsp[-3].V_int));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[-5]).first_line);
   delete (yyvsp[-5].P_string);
   delete (yyvsp[0].P_string);
}
#line 4083 "mdl.tab.c"
    break;

  case 176: /* nameComment: IDENTIFIER '(' INT_CONSTANT ',' INT_CONSTANT ')' ':' STRING_LITERAL  */
#line 1197 "mdl.y"
                                                                       {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[-7].P_string), *(yyvsp[0].P_string), (yyvsp[-5].V_int), (yyvsp[-3].V_int));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[-7]).first_line);
   delete (yyvsp[-7].P_string);
   delete (yyvsp[0].P_string);
}
#line 4094 "mdl.tab.c"
    break;

  case 177: /* nameCommentList: nameComment  */
#line 1205 "mdl.y"
                             {   
   (yyval.P_nameCommentList) = new C_nameCommentList((yyvsp[0].P_nameComment));
   (yyval.P_nameCommentList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4103 "mdl.tab.c"
    break;

  case 178: /* nameCommentList: nameCommentList ',' nameComment  */
#line 1209 "mdl.y"
                                  {
   (yyval.P_nameCommentList) = new C_nameCommentList((yyvsp[-2].P_nameCommentList), (yyvsp[0].P_nameComment));   
   (yyval.P_nameCommentList)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 4112 "mdl.tab.c"
    break;

  case 179: /* nameCommentArgument: IDENTIFIER  */
#line 1215 "mdl.y"
                                {
   (yyval.P_nameComment) = new C_nameComment(*(yyvsp[0].P_string));
   (yyval.P_nameComment)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
   delete (yyvsp[0].P_string);
}
#line 4122 "mdl.tab.c"
    break;

  case 180: /* nameCommentArgumentList: nameCommentArgument  */
#line 1222 "mdl.y"
                                             {
   (yyval.P_nameCommentList) = new C_nameCommentList((yyvsp[0].P_nameComment));
   (yyval.P_nameCommentList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4131 "mdl.tab.c"
    break;

  case 181: /* functor: FUNCTOR IDENTIFIER '{' functorStatementList '}'  */
#line 1228 "mdl.y"
                                                         {
    (yyval.P_functor) = new C_functor(*(yyvsp[-3].P_string), (yyvsp[-1].P_generalList));
    (yyval.P_functor)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);
    delete (yyvsp[-3].P_string);
}
#line 4141 "mdl.tab.c"
    break;

  case 182: /* functor: FRAMEWORK FUNCTOR IDENTIFIER '{' functorStatementList '}'  */
#line 1233 "mdl.y"
                                                            {
    (yyval.P_functor) = new C_functor(*(yyvsp[-3].P_string), (yyvsp[-1].P_generalList));
    (yyval.P_functor)->setTokenLocation(CURRENTFILE, (yylsp[-5]).first_line);
    (yyval.P_functor)->setFrameWorkElement();
    delete (yyvsp[-3].P_string);
}
#line 4152 "mdl.tab.c"
    break;

  case 183: /* functor: FUNCTOR IDENTIFIER CATEGORY STRING_LITERAL '{' functorStatementList '}'  */
#line 1239 "mdl.y"
                                                                          {
    (yyval.P_functor) = new C_functor(*(yyvsp[-5].P_string), (yyvsp[-1].P_generalList), *(yyvsp[-3].P_string));
    (yyval.P_functor)->setTokenLocation(CURRENTFILE, (yylsp[-6]).first_line);
    delete (yyvsp[-5].P_string);
    delete (yyvsp[-3].P_string);
}
#line 4163 "mdl.tab.c"
    break;

  case 184: /* functor: FRAMEWORK FUNCTOR IDENTIFIER CATEGORY STRING_LITERAL '{' functorStatementList '}'  */
#line 1245 "mdl.y"
                                                                                    {
    (yyval.P_functor) = new C_functor(*(yyvsp[-5].P_string), (yyvsp[-1].P_generalList), *(yyvsp[-3].P_string));
    (yyval.P_functor)->setTokenLocation(CURRENTFILE, (yylsp[-7]).first_line);
    (yyval.P_functor)->setFrameWorkElement();
    delete (yyvsp[-5].P_string);
    delete (yyvsp[-3].P_string);
}
#line 4175 "mdl.tab.c"
    break;

  case 185: /* functorStatementList: functorStatement  */
#line 1254 "mdl.y"
                                       {
   (yyval.P_generalList) = new C_generalList((yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4184 "mdl.tab.c"
    break;

  case 186: /* functorStatementList: functorStatementList functorStatement  */
#line 1258 "mdl.y"
                                        {
   (yyval.P_generalList) = new C_generalList((yyvsp[-1].P_generalList), (yyvsp[0].P_general));   
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 4193 "mdl.tab.c"
    break;

  case 187: /* functorStatement: noop  */
#line 1264 "mdl.y"
                       {
   (yyval.P_general) = (yyvsp[0].P_noop);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4202 "mdl.tab.c"
    break;

  case 188: /* functorStatement: execute  */
#line 1268 "mdl.y"
          {
   (yyval.P_general) = (yyvsp[0].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4211 "mdl.tab.c"
    break;

  case 189: /* functorStatement: initialize  */
#line 1272 "mdl.y"
             {
   (yyval.P_general) = (yyvsp[0].P_general);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4220 "mdl.tab.c"
    break;

  case 190: /* execute: returnType EXECUTE '(' ')' ';'  */
#line 1278 "mdl.y"
                                        {
   (yyval.P_general) = new C_execute((yyvsp[-4].P_returnType));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);  
}
#line 4229 "mdl.tab.c"
    break;

  case 191: /* execute: returnType EXECUTE '(' ELLIPSIS ')' ';'  */
#line 1282 "mdl.y"
                                          {
   (yyval.P_general) = new C_execute((yyvsp[-5].P_returnType), true);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-5]).first_line);  
}
#line 4238 "mdl.tab.c"
    break;

  case 192: /* execute: returnType EXECUTE '(' argumentDataTypeList ')' ';'  */
#line 1286 "mdl.y"
                                                      {
   (yyval.P_general) = new C_execute((yyvsp[-5].P_returnType), (yyvsp[-2].P_generalList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-5]).first_line);  
}
#line 4247 "mdl.tab.c"
    break;

  case 193: /* execute: returnType EXECUTE '(' argumentDataTypeList ',' ELLIPSIS ')' ';'  */
#line 1290 "mdl.y"
                                                                   {
   (yyval.P_general) = new C_execute((yyvsp[-7].P_returnType), (yyvsp[-4].P_generalList), true);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-7]).first_line);  
}
#line 4256 "mdl.tab.c"
    break;

  case 194: /* returnType: VOID  */
#line 1296 "mdl.y"
                 {
   (yyval.P_returnType) = new C_returnType(true);
   (yyval.P_returnType)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);    
}
#line 4265 "mdl.tab.c"
    break;

  case 195: /* returnType: typeClassifier  */
#line 1300 "mdl.y"
                 {
   (yyval.P_returnType) = new C_returnType((yyvsp[0].P_typeClassifier));
   (yyval.P_returnType)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);    
}
#line 4274 "mdl.tab.c"
    break;

  case 196: /* initialize: INITIALIZE '(' ')' ';'  */
#line 1306 "mdl.y"
                                   {
   (yyval.P_general) = new C_initialize();
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);  
}
#line 4283 "mdl.tab.c"
    break;

  case 197: /* initialize: INITIALIZE '(' ELLIPSIS ')' ';'  */
#line 1310 "mdl.y"
                                  {
   (yyval.P_general) = new C_initialize(true);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);  
}
#line 4292 "mdl.tab.c"
    break;

  case 198: /* initialize: INITIALIZE '(' argumentDataTypeList ')' ';'  */
#line 1314 "mdl.y"
                                              {
   (yyval.P_general) = new C_initialize((yyvsp[-2].P_generalList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-4]).first_line);  
}
#line 4301 "mdl.tab.c"
    break;

  case 199: /* initialize: INITIALIZE '(' argumentDataTypeList ',' ELLIPSIS ')' ';'  */
#line 1318 "mdl.y"
                                                           {
   (yyval.P_general) = new C_initialize((yyvsp[-4].P_generalList), true);
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-6]).first_line);  
}
#line 4310 "mdl.tab.c"
    break;

  case 200: /* argumentDataTypeList: argumentDataType  */
#line 1324 "mdl.y"
                                       {
   (yyval.P_generalList) = new C_generalList((yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4319 "mdl.tab.c"
    break;

  case 201: /* argumentDataTypeList: argumentDataTypeList ',' argumentDataType  */
#line 1328 "mdl.y"
                                            {
   (yyval.P_generalList) = new C_generalList((yyvsp[-2].P_generalList), (yyvsp[0].P_general));
   (yyval.P_generalList)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 4328 "mdl.tab.c"
    break;

  case 202: /* argumentDataType: typeClassifier nameCommentArgumentList  */
#line 1334 "mdl.y"
                                                         {
   (yyval.P_general) = new C_dataType((yyvsp[-1].P_typeClassifier), (yyvsp[0].P_nameCommentList));
   (yyval.P_general)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 4337 "mdl.tab.c"
    break;

  case 203: /* triggeredFunctionInstance: TRIGGEREDFUNCTION identifierList ';'  */
#line 1340 "mdl.y"
                                                                {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionInstance((yyvsp[-1].P_identifierList), TriggeredFunction::_SERIAL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 4346 "mdl.tab.c"
    break;

  case 204: /* triggeredFunctionInstance: SERIAL TRIGGEREDFUNCTION identifierList ';'  */
#line 1344 "mdl.y"
                                              {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionInstance((yyvsp[-1].P_identifierList), TriggeredFunction::_SERIAL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 4355 "mdl.tab.c"
    break;

  case 205: /* triggeredFunctionInstance: PARALLEL TRIGGEREDFUNCTION identifierList ';'  */
#line 1348 "mdl.y"
                                                {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionInstance((yyvsp[-1].P_identifierList), TriggeredFunction::_PARALLEL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 4364 "mdl.tab.c"
    break;

  case 206: /* triggeredFunctionShared: TRIGGEREDFUNCTION identifierList ';'  */
#line 1354 "mdl.y"
                                                              {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionShared((yyvsp[-1].P_identifierList), TriggeredFunction::_SERIAL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 4373 "mdl.tab.c"
    break;

  case 207: /* triggeredFunctionShared: SERIAL TRIGGEREDFUNCTION identifierList ';'  */
#line 1358 "mdl.y"
                                              {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionShared((yyvsp[-1].P_identifierList), TriggeredFunction::_SERIAL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 4382 "mdl.tab.c"
    break;

  case 208: /* triggeredFunctionShared: PARALLEL TRIGGEREDFUNCTION identifierList ';'  */
#line 1362 "mdl.y"
                                                {
   (yyval.P_triggeredFunction) = new C_triggeredFunctionShared((yyvsp[-1].P_identifierList), TriggeredFunction::_PARALLEL);
   (yyval.P_triggeredFunction)->setTokenLocation(CURRENTFILE, (yylsp[-3]).first_line);
}
#line 4391 "mdl.tab.c"
    break;

  case 209: /* nonPointerTypeClassifier: typeCore  */
#line 1373 "mdl.y"
                                   {
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[0].P_typeCore));
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4400 "mdl.tab.c"
    break;

  case 210: /* nonPointerTypeClassifier: array  */
#line 1377 "mdl.y"
        { 
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[0].P_array));
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4409 "mdl.tab.c"
    break;

  case 211: /* typeClassifier: typeCore  */
#line 1383 "mdl.y"
                         {
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[0].P_typeCore));
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4418 "mdl.tab.c"
    break;

  case 212: /* typeClassifier: typeCore STAR  */
#line 1387 "mdl.y"
                {
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[-1].P_typeCore), true);
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 4427 "mdl.tab.c"
    break;

  case 213: /* typeClassifier: array  */
#line 1391 "mdl.y"
        { 
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[0].P_array));
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4436 "mdl.tab.c"
    break;

  case 214: /* typeClassifier: array STAR  */
#line 1395 "mdl.y"
             {
   (yyval.P_typeClassifier) = new C_typeClassifier((yyvsp[-1].P_array), true);
   (yyval.P_typeClassifier)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 4445 "mdl.tab.c"
    break;

  case 215: /* typeCore: STRING  */
#line 1401 "mdl.y"
                 {
   (yyval.P_typeCore) = new C_typeCore(new StringType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4454 "mdl.tab.c"
    break;

  case 216: /* typeCore: BOOL  */
#line 1405 "mdl.y"
       {
   (yyval.P_typeCore) = new C_typeCore(new BoolType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4463 "mdl.tab.c"
    break;

  case 217: /* typeCore: CHAR  */
#line 1409 "mdl.y"
       {
   (yyval.P_typeCore) = new C_typeCore(new CharType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4472 "mdl.tab.c"
    break;

  case 218: /* typeCore: SHORT  */
#line 1413 "mdl.y"
        {
   (yyval.P_typeCore) = new C_typeCore(new ShortType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4481 "mdl.tab.c"
    break;

  case 219: /* typeCore: INT  */
#line 1417 "mdl.y"
      {
   (yyval.P_typeCore) = new C_typeCore(new IntType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4490 "mdl.tab.c"
    break;

  case 220: /* typeCore: LONG  */
#line 1421 "mdl.y"
       {
   (yyval.P_typeCore) = new C_typeCore(new LongType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4499 "mdl.tab.c"
    break;

  case 221: /* typeCore: FLOAT  */
#line 1425 "mdl.y"
        {
   (yyval.P_typeCore) = new C_typeCore(new FloatType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4508 "mdl.tab.c"
    break;

  case 222: /* typeCore: DOUBLE  */
#line 1429 "mdl.y"
         {
   (yyval.P_typeCore) = new C_typeCore(new DoubleType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4517 "mdl.tab.c"
    break;

  case 223: /* typeCore: LONG DOUBLE  */
#line 1433 "mdl.y"
              {
   (yyval.P_typeCore) = new C_typeCore(new LongDoubleType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[-1]).first_line);
}
#line 4526 "mdl.tab.c"
    break;

  case 224: /* typeCore: UNSIGNED  */
#line 1437 "mdl.y"
           {
   (yyval.P_typeCore) = new C_typeCore(new UnsignedType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4535 "mdl.tab.c"
    break;

  case 225: /* typeCore: EDGE  */
#line 1441 "mdl.y"
       {
   (yyval.P_typeCore) = new C_typeCore(new EdgeType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4544 "mdl.tab.c"
    break;

  case 226: /* typeCore: EDGESET  */
#line 1445 "mdl.y"
          {
   (yyval.P_typeCore) = new C_typeCore(new EdgeSetType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4553 "mdl.tab.c"
    break;

  case 227: /* typeCore: EDGETYPE  */
#line 1449 "mdl.y"
           {
   (yyval.P_typeCore) = new C_typeCore(new EdgeTypeType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4562 "mdl.tab.c"
    break;

  case 228: /* typeCore: FUNCTOR  */
#line 1453 "mdl.y"
          {
   (yyval.P_typeCore) = new C_typeCore(new FunctorType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4571 "mdl.tab.c"
    break;

  case 229: /* typeCore: GRID  */
#line 1457 "mdl.y"
       {
   (yyval.P_typeCore) = new C_typeCore(new GridType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4580 "mdl.tab.c"
    break;

  case 230: /* typeCore: NODE  */
#line 1461 "mdl.y"
       {
   (yyval.P_typeCore) = new C_typeCore(new NodeType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4589 "mdl.tab.c"
    break;

  case 231: /* typeCore: NODESET  */
#line 1465 "mdl.y"
          {
   (yyval.P_typeCore) = new C_typeCore(new NodeSetType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4598 "mdl.tab.c"
    break;

  case 232: /* typeCore: NODETYPE  */
#line 1469 "mdl.y"
           {
   (yyval.P_typeCore) = new C_typeCore(new NodeTypeType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4607 "mdl.tab.c"
    break;

  case 233: /* typeCore: SERVICE  */
#line 1473 "mdl.y"
          {
   (yyval.P_typeCore) = new C_typeCore(new ServiceType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4616 "mdl.tab.c"
    break;

  case 234: /* typeCore: REPERTOIRE  */
#line 1477 "mdl.y"
             {
   (yyval.P_typeCore) = new C_typeCore(new RepertoireType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4625 "mdl.tab.c"
    break;

  case 235: /* typeCore: TRIGGER  */
#line 1481 "mdl.y"
          {
   (yyval.P_typeCore) = new C_typeCore(new TriggerType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4634 "mdl.tab.c"
    break;

  case 236: /* typeCore: PARAMETERSET  */
#line 1485 "mdl.y"
               {
   (yyval.P_typeCore) = new C_typeCore(new ParameterSetType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4643 "mdl.tab.c"
    break;

  case 237: /* typeCore: NDPAIRLIST  */
#line 1489 "mdl.y"
             {
   (yyval.P_typeCore) = new C_typeCore(new NDPairListType());
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
}
#line 4652 "mdl.tab.c"
    break;

  case 238: /* typeCore: IDENTIFIER  */
#line 1493 "mdl.y"
             {
   (yyval.P_typeCore) = new C_typeCore(*(yyvsp[0].P_string));
   (yyval.P_typeCore)->setTokenLocation(CURRENTFILE, (yylsp[0]).first_line);
   delete (yyvsp[0].P_string);
}
#line 4662 "mdl.tab.c"
    break;

  case 239: /* array: typeClassifier '[' ']'  */
#line 1500 "mdl.y"
                               {
   (yyval.P_array) = new C_array((yyvsp[-2].P_typeClassifier));
   (yyval.P_array)->setTokenLocation(CURRENTFILE, (yylsp[-2]).first_line);
}
#line 4671 "mdl.tab.c"
    break;


#line 4675 "mdl.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == MDLEMPTY ? symbol_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      {
        yypcontext_t yyctx
          = {yyssp, yytoken, &yylloc};
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == -1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = YY_CAST (char *,
                             YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
            if (yymsg)
              {
                yysyntax_error_status
                  = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
                yymsgp = yymsg;
              }
            else
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = YYENOMEM;
              }
          }
        yyerror (&yylloc, parm, yymsgp);
        if (yysyntax_error_status == YYENOMEM)
          YYNOMEM;
      }
    }

  yyerror_range[1] = yylloc;
  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= MDLEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == MDLEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc, parm);
          yychar = MDLEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
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
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += symbol_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == symbol_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, yylsp, parm);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  ++yylsp;
  YYLLOC_DEFAULT (*yylsp, yyerror_range, 2);

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (&yylloc, parm, YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != MDLEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc, parm);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, yylsp, parm);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
  return yyresult;
}

#line 1506 "mdl.y"


// Lexer implementation
int mdllex(YYSTYPE* lvalp, YYLTYPE* locp, void* parm)
{
    return CONTEXT->_lexer->lex(lvalp, locp, CONTEXT);
}

// Error handling function
void mdlerror(YYLTYPE* locp, void* parm, const char* s)
{
    MdlContext *c = (MdlContext *) parm;
    if (c && c->_lexer && locp) {
        // Store the error location in the context
        c->setLastError(c->_lexer->currentFileName, locp->first_line);
        
        // Optional: add more detailed diagnostics for debugging
        if (!c->isErrorDisplayed()) {
            fprintf(stderr, "ERROR: [%s:%d] %s\n", 
                    c->_lexer->currentFileName.c_str(),
                    locp->first_line, 
                    s);
            
            // When encountering a "missing semicolon" error, suggest checking the previous line
            if (strstr(s, "expecting ';'") != NULL) {
                fprintf(stderr, "HINT: The error might be a missing semicolon at the end of line %d.\n", 
                  locp->first_line - 1);
            }
        }
    } else {
        fprintf(stderr, "Parser error: %s (location info unavailable)\n", s);
    }
}

// Main and Helper Functions 

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
  std::cout << ". . . . . . . . . . . . . . . . . . . . . ." << std::endl
            << ".                                         ." << std::endl
            << ".        /\\/\\/\\    ______     _           ." << std::endl
            << ".       /      \\  |  __  \\   | |          ." << std::endl
            << ".      /  /\\/\\  \\ | |  | |   | |          ." << std::endl
            << ".     /  /    \\  \\| |  | |   | |          ." << std::endl
            << ".    /  /      \\  \\ |__| |   | | ___      ." << std::endl
            << ".    \\/\\/      /\\/ \\____/    |_|____\\     ." << std::endl
            << ".     \\  \\    /     \\   /        /        ." << std::endl
            << ".      \\  \\  /       \\ /        /         ." << std::endl
            << ".       \\  \\/         \\        /          ." << std::endl
            << ".        \\  \\\\        /\\      /           ." << std::endl
            << ".         \\  \\\\      /  \\    /            ." << std::endl
            << ".          \\  \\\\    /    \\  /             ." << std::endl
            << ".           \\  \\\\  /      \\/              ." << std::endl
            << ".            \\  \\\\/        *              ." << std::endl
            << ".             \\  \\\\       * *             ." << std::endl
            << ".              \\__/      * * *            ." << std::endl
            << ". . . . . . . . . . . . . . . . . . . . . .\n\n\n";

   Initializer init(argc, argv);
   if (init.execute()) {
      return 0;
   } else {
      return 1;
   }
}
