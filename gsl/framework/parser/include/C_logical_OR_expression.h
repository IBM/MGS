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
