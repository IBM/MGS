/**
 * bison_compat.h
 * 
 * Compatibility layer for modern Bison (3.8.2) in legacy codebase
 */

#ifndef BISON_COMPAT_H
#define BISON_COMPAT_H

#include <limits.h>

// Let Bison define YYSTYPE and yysymbol_kind_t
// We only provide fallbacks for older code

// Only define YYLTYPE if it's not already defined elsewhere
#if !defined(YYLTYPE_DEFINED) && !defined(YYLTYPE)
#define YYLTYPE_DEFINED
typedef struct YYLTYPE {
  int first_line;
  int first_column;
  int last_line;
  int last_column;
} YYLTYPE;
#endif

// Null pointer compatibility
#ifndef YY_NULLPTR
#if defined __cplusplus && 201103L <= __cplusplus
#define YY_NULLPTR nullptr
#else
#define YY_NULLPTR 0
#endif
#endif

// Memory allocation compatibility
#ifndef YYPTRDIFF_T
#define YYPTRDIFF_T long int
#endif

#ifndef YYPTRDIFF_MAXIMUM
#define YYPTRDIFF_MAXIMUM LONG_MAX
#endif

// For size_t compatibility
#ifndef YYSIZE_T
#ifdef __SIZE_TYPE__
#define YYSIZE_T __SIZE_TYPE__
#elif defined size_t
#define YYSIZE_T size_t
#elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#include <stddef.h> // For size_t
#define YYSIZE_T size_t
#else
#define YYSIZE_T unsigned int
#endif
#endif

#ifndef YYSIZE_MAX
#define YYSIZE_MAX SIZE_MAX
#endif

// Stack allocation maximum
#ifndef YYSTACK_ALLOC_MAXIMUM
#define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAX
#endif

// Cast macros for compatibility
#ifndef YY_CAST
#define YY_CAST(Type, Val) ((Type) (Val))
#define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#endif

#endif // BISON_COMPAT_H
