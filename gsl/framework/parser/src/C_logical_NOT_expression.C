// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_logical_NOT_expression.h"
#include "C_primary_expression.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "SyntaxError.h"
#include "C_production_grid.h"
#include <algorithm>

void C_logical_NOT_expression::internalExecute(GslContext *c, Grid* g)
{
   _primaryExpression->execute(c, g);

   _layers.clear();
   std::insert_iterator<std::list<GridLayerDescriptor*> > layers_iter(
      _layers, _layers.begin());


   std::list<GridLayerDescriptor*> lin;
   const std::vector<GridLayerDescriptor*>& glayers = g->getLayers();
   std::vector<GridLayerDescriptor*>::const_iterator iter, end = glayers.end();
   for (iter = glayers.begin(); iter != end; ++iter) {
      lin.push_back(*iter);
   }
   lin.sort();
   std::list<GridLayerDescriptor*>::iterator lin_begin = lin.begin(), 
      lin_end = lin.end();

   std::list<GridLayerDescriptor*> lin2 = _primaryExpression->getLayers();
   lin2.sort();
   std::list<GridLayerDescriptor*>::iterator lin2_begin = lin2.begin(), 
      lin2_end = lin2.end();

   set_difference(lin_begin, lin_end, lin2_begin, lin2_end, layers_iter);

   _layers.unique();
}


const std::list<GridLayerDescriptor*>& 
C_logical_NOT_expression::getLayers() const
{
   return _layers;
}


C_logical_NOT_expression::C_logical_NOT_expression(
   const C_logical_NOT_expression& rv)
   : C_production_grid(rv), _layers(rv._layers), _primaryExpression(0)
{
   if (rv._primaryExpression) {
      _primaryExpression = rv._primaryExpression->duplicate();
   }
}


C_logical_NOT_expression::C_logical_NOT_expression(
   C_primary_expression *pe, SyntaxError * error)
   : C_production_grid(error), _primaryExpression(pe)
{
}


C_logical_NOT_expression* C_logical_NOT_expression::duplicate() const
{
   return new C_logical_NOT_expression(*this);
}


C_logical_NOT_expression::~C_logical_NOT_expression()
{
   delete _primaryExpression;
}

void C_logical_NOT_expression::checkChildren() 
{
   if (_primaryExpression) {
      _primaryExpression->checkChildren();
      if (_primaryExpression->isError()) {
         setError();
      }
   }
} 

void C_logical_NOT_expression::recursivePrint() 
{
   if (_primaryExpression) {
      _primaryExpression->recursivePrint();
   }
   printErrorMessage();
} 
