// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_equality_expression_H
#define C_equality_expression_H
#include "Copyright.h"

#include <string>
#include <list>
#include "C_production_grid.h"

class C_primary_expression;
class C_name;
class LensContext;
class GridLayerDescriptor;
class Grid;
class SyntaxError;

class C_equality_expression : public C_production_grid
{
   public:
      C_equality_expression(const C_equality_expression&);
      C_equality_expression(C_primary_expression *, SyntaxError * error);
      C_equality_expression(C_name *, std::string *, bool, 
			    SyntaxError * error);
      virtual ~C_equality_expression();
      virtual C_equality_expression* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      bool _equivalence;
      std::list<GridLayerDescriptor*> _layers;
      C_name* _name;
      C_primary_expression* _primaryExpression;
      std::string* _value;
};
#endif
