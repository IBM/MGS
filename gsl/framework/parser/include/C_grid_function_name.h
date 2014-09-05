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

#ifndef C_grid_function_name_H
#define C_grid_function_name_H
#include "Copyright.h"

#include <string>
#include "C_production_grid.h"

class C_argument_list;
class LensContext;
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
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      Type _type;
      C_argument_list *_argList;
      void initNodes(LensContext *c, Grid* g);
      void layers(LensContext *c, Grid* g);
      C_declarator *_declarator;
      C_grid_granule_volume *_gridGranule;
};
#endif
