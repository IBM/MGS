// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_default_clause_H
#define C_default_clause_H
#include "Copyright.h"

#include "C_production_adi.h"

class C_constant;
class GslContext;
class ArrayDataItem;
class SyntaxError;

class C_default_clause : public C_production_adi
{
   public:
      C_default_clause(const C_default_clause&);
      C_default_clause(C_constant *, SyntaxError *);
      virtual ~C_default_clause();
      virtual C_default_clause* duplicate() const;
      virtual void internalExecute(GslContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_constant * getConstant() const {
	 return _constant;
      }

   private:
      C_constant* _constant;
};
#endif
