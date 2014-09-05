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
