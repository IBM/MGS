// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _C_COMPLEX_FUNCTOR_CLAUSE_H_
#define _C_COMPLEX_FUNCTOR_CLAUSE_H_
#include "Copyright.h"

#include "C_production.h"

class LensContext;

class C_complex_functor_clause : public C_production
{
   public:
      enum Type{_CONSTRUCTOR, _FUNCTION, _RETURN};
      C_complex_functor_clause(const C_complex_functor_clause&);
      C_complex_functor_clause(Type t, SyntaxError* error);
      virtual ~C_complex_functor_clause();
      virtual C_complex_functor_clause* duplicate() const = 0;
      virtual void internalExecute(LensContext *) = 0;
      virtual void checkChildren();
      virtual void recursivePrint();
      Type getType() const {
	 return _type;
      }

   private:
      Type _type;
};
#endif
