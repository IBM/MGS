// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_LOGICAL_NOT_EXPRESSION
#define C_LOGICAL_NOT_EXPRESSION
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_primary_expression;
class Grid;
class GslContext;
class GridLayerDescriptor;
class SyntaxError;

class C_logical_NOT_expression : public C_production_grid
{
   public:
      C_logical_NOT_expression(const C_logical_NOT_expression&);
      C_logical_NOT_expression(C_primary_expression *, SyntaxError *);
      virtual void internalExecute(GslContext *, Grid *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual C_logical_NOT_expression* duplicate() const;
      const std::list<GridLayerDescriptor*>& getLayers() const;
      virtual ~C_logical_NOT_expression();

   private:
      std::list<GridLayerDescriptor*> _layers;
      C_primary_expression* _primaryExpression;
};
#endif
