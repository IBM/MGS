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

#ifndef C_primary_expression_H
#define C_primary_expression_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_layer_set;
class C_logical_OR_expression;
class C_logical_NOT_expression;
class LensContext;
class Grid;
class GridLayerDescriptor;
class SyntaxError;

class C_primary_expression : public C_production_grid
{
   public:
      C_primary_expression(const C_primary_expression&);
      C_primary_expression(C_layer_set *, SyntaxError *);
      C_primary_expression(C_logical_OR_expression *, SyntaxError *);
      C_primary_expression(C_logical_NOT_expression *, SyntaxError *);
      virtual ~C_primary_expression();
      virtual C_primary_expression* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      C_layer_set* _layerSet;
      C_logical_NOT_expression* _logicalNotExpression;
      C_logical_OR_expression* _logicalOrExpression;
      std::list<GridLayerDescriptor*> _layers;
};
#endif
