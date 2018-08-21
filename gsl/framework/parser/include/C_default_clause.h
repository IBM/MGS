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

#ifndef C_default_clause_H
#define C_default_clause_H
#include "Copyright.h"

#include "C_production_adi.h"

class C_constant;
class LensContext;
class ArrayDataItem;
class SyntaxError;

class C_default_clause : public C_production_adi
{
   public:
      C_default_clause(const C_default_clause&);
      C_default_clause(C_constant *, SyntaxError *);
      virtual ~C_default_clause();
      virtual C_default_clause* duplicate() const;
      virtual void internalExecute(LensContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_constant * getConstant() const {
	 return _constant;
      }

   private:
      C_constant* _constant;
};
#endif
