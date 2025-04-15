// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_grid_function_specifier_H
#define C_grid_function_specifier_H
#include "Copyright.h"

#include "C_production_grid.h"

class C_directive;
class C_grid_function_name;
class LensContext;
class Grid;
class SyntaxError;

class C_grid_function_specifier : public C_production_grid
{
   public:
      C_grid_function_specifier(const C_grid_function_specifier&);
      C_grid_function_specifier(C_directive *, SyntaxError *);
      C_grid_function_specifier(C_grid_function_name *, SyntaxError *);
      virtual ~C_grid_function_specifier();
      virtual C_grid_function_specifier* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_directive* _directive;
      C_grid_function_name* _gridFuncName;

};
#endif
