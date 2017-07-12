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

#ifndef C_production_grid_H
#define C_production_grid_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
class SyntaxError;
class Grid;

class C_production_grid : public C_production
{
   public:
      C_production_grid(SyntaxError* error);
      C_production_grid(const C_production_grid&);
      virtual ~C_production_grid();
      virtual C_production_grid* duplicate() const = 0;
      virtual void execute(LensContext *, Grid *);
      virtual void checkChildren() {};
      virtual void recursivePrint() {};
   protected:
      virtual void internalExecute(LensContext *);
      virtual void internalExecute(LensContext *, Grid *) = 0;
};
#endif
