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
