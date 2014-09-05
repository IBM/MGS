// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_production_adi_H
#define C_production_adi_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
class SyntaxError;
class ArrayDataItem;

class C_production_adi : public C_production
{
   public:
      C_production_adi(SyntaxError* error);
      C_production_adi(const C_production_adi&);
      virtual ~C_production_adi();
      virtual C_production_adi* duplicate() const = 0;
      virtual void execute(LensContext *);
      virtual void execute(LensContext *, ArrayDataItem *);
      virtual void checkChildren() {};
      virtual void recursivePrint() {};
   protected:
      virtual void internalExecute(LensContext *);
      virtual void internalExecute(LensContext *, ArrayDataItem *) = 0;
};
#endif
