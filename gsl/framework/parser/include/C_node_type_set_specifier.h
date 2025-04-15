// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_node_type_set_specifier_H
#define C_node_type_set_specifier_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_node_type_set_specifier_clause;
class Grid;
class GridLayerDescriptor;
class LensContext;
class SyntaxError;

class C_node_type_set_specifier : public C_production_grid
{
   public:
      C_node_type_set_specifier(const C_node_type_set_specifier&);
      C_node_type_set_specifier(C_node_type_set_specifier_clause *, 
				SyntaxError *);
      virtual ~C_node_type_set_specifier ();
      virtual C_node_type_set_specifier* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      C_node_type_set_specifier_clause* _nodeTypeSetSpecifierClause;
      std::list<GridLayerDescriptor*> _layers;
};
#endif
