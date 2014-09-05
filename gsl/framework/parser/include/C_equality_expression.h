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
