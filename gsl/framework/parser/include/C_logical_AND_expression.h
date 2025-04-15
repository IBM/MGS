// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_logical_AND_expression_H
#define C_logical_AND_expression_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_equality_expression;
class LensContext;
class GridLayerDescriptor;
class Grid;
class SyntaxError;

class C_logical_AND_expression : public C_production_grid
{
   public:
      C_logical_AND_expression(const C_logical_AND_expression&);
      C_logical_AND_expression(C_logical_AND_expression *, 
	 C_equality_expression *, SyntaxError *);
      C_logical_AND_expression(C_equality_expression *, SyntaxError *);
      std::list<C_equality_expression*>* releaseSet();
      virtual ~C_logical_AND_expression();
      virtual C_logical_AND_expression* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      std::list<C_equality_expression*>* _listEqualityExpression;
      std::list<GridLayerDescriptor*> _layers;
};
#endif
