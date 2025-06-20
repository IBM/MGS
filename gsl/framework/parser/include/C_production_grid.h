// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_production_grid_H
#define C_production_grid_H
#include "Copyright.h"

#include "C_production.h"

class GslContext;
class SyntaxError;
class Grid;

class C_production_grid : public C_production
{
   public:
      using C_production::execute; // tell the compiler we want both from base and overloaded
      C_production_grid(SyntaxError* error);
      C_production_grid(const C_production_grid&);
      virtual ~C_production_grid();
      virtual C_production_grid* duplicate() const = 0;
      virtual void execute(GslContext *, Grid *);
      virtual void checkChildren() {};
      virtual void recursivePrint() {};
   protected:
      virtual void internalExecute(GslContext *);
      virtual void internalExecute(GslContext *, Grid *) = 0;
};
#endif
