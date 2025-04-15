// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_logical_OR_expression_H
#define C_logical_OR_expression_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_logical_AND_expression;
class GridLayerDescriptor;
class Grid;
class LensContext;
class SyntaxError;

class C_logical_OR_expression : public C_production_grid
{
   public:
      C_logical_OR_expression(const C_logical_OR_expression&);
      C_logical_OR_expression(C_logical_AND_expression *, SyntaxError *);
      C_logical_OR_expression(C_logical_OR_expression *, 
	 C_logical_AND_expression *, SyntaxError *);
      virtual ~C_logical_OR_expression ();
      std::list<C_logical_AND_expression*>* releaseSet();
      virtual C_logical_OR_expression* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      std::list<C_logical_AND_expression*>* _listLogicalAndExpression;
      std::list<GridLayerDescriptor*> _layers;
};
#endif
