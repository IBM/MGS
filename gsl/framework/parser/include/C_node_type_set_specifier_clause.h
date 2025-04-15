// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_node_type_set_specifier_clause_H
#define C_node_type_set_specifier_clause_H
#include "Copyright.h"

#include <list>
#include "NodeTypeSet.h"
#include "C_production.h"

class C_layer_set;
class C_logical_OR_expression;
class Grid;
class GridLayerDescriptor;
class LensContext;
class SyntaxError;

class C_node_type_set_specifier_clause : public NodeTypeSet, public C_production
{
   public:
      C_node_type_set_specifier_clause(
	 const C_node_type_set_specifier_clause&);
      C_node_type_set_specifier_clause(C_layer_set *, SyntaxError *);
      C_node_type_set_specifier_clause(C_logical_OR_expression *, 
				       SyntaxError *);
      virtual ~C_node_type_set_specifier_clause ();
      virtual C_node_type_set_specifier_clause* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers(Grid* g);

   private:
      C_layer_set* _layerSet;
      C_logical_OR_expression* _logicalOrExpression;
      LensContext* _storedContext;
      Grid* _lastGrid;
};
#endif
