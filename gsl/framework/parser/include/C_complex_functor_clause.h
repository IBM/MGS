// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
