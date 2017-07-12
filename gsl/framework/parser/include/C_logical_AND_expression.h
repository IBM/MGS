// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
