// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_LOGICAL_NOT_EXPRESSION
#define C_LOGICAL_NOT_EXPRESSION
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_primary_expression;
class Grid;
class LensContext;
class GridLayerDescriptor;
class SyntaxError;

class C_logical_NOT_expression : public C_production_grid
{
   public:
      C_logical_NOT_expression(const C_logical_NOT_expression&);
      C_logical_NOT_expression(C_primary_expression *, SyntaxError *);
      virtual void internalExecute(LensContext *, Grid *);
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
