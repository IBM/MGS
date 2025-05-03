// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_grid_function_name_H
#define C_grid_function_name_H
#include "Copyright.h"

#include <string>
#include "C_production_grid.h"

class C_argument_list;
class GslContext;
class Grid;
class C_declarator;
class C_grid_granule_volume;
class SyntaxError;

class C_grid_function_name : public C_production_grid
{
   public:
      enum Type { _INITNODES, _LAYER };
      C_grid_function_name(const C_grid_function_name&);
      C_grid_function_name(C_argument_list *, SyntaxError *);
      C_grid_function_name(C_declarator *, C_argument_list *, SyntaxError *);
      virtual ~C_grid_function_name();
      virtual C_grid_function_name* duplicate() const;
      virtual void internalExecute(GslContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      Type _type;
      /* representing 4 or 5 arguments passed to 'Layer' statement in GSL */
      C_argument_list *_argList;
      void initNodes(GslContext *c, Grid* g);
      void layers(GslContext *c, Grid* g);
      C_declarator *_declarator;
      C_grid_granule_volume *_gridGranule;
};
#endif
