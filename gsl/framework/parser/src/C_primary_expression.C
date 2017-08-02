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

#include "C_primary_expression.h"
#include "C_logical_OR_expression.h"
#include "C_logical_NOT_expression.h"
#include "C_layer_set.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "SyntaxError.h"
#include "C_production_grid.h"

void C_primary_expression::internalExecute(LensContext *c, Grid* g)
{
   if (_layerSet) {
      _layerSet->execute(c, g);
      _layers = _layerSet->getLayers();
   }
   if (_logicalNotExpression) {
      _logicalNotExpression->execute(c, g);
      _layers = _logicalNotExpression->getLayers();
   }
   if (_logicalOrExpression) {
      _logicalOrExpression->execute(c, g);
      _layers = _logicalOrExpression->getLayers();
   }
}


const std::list<GridLayerDescriptor*>& C_primary_expression::getLayers() const
{
   return _layers;
}


C_primary_expression::C_primary_expression(const C_primary_expression& rv)
   : C_production_grid(rv), _layerSet(0), _logicalNotExpression(0), 
     _logicalOrExpression(0), _layers(rv._layers)
{
   if (rv._layerSet) {
      _layerSet = rv._layerSet->duplicate();
   }
   if (rv._logicalNotExpression) {
      _logicalNotExpression = rv._logicalNotExpression->duplicate();
   }
   if (rv._logicalOrExpression) {
      _logicalOrExpression = rv._logicalOrExpression->duplicate();
   }
}


C_primary_expression::C_primary_expression(
   C_logical_OR_expression *loe, SyntaxError * error)
   : C_production_grid(error), _layerSet(0), _logicalNotExpression(0), 
     _logicalOrExpression(loe)
{
}


C_primary_expression::C_primary_expression(
   C_logical_NOT_expression *lne, SyntaxError * error)
   : C_production_grid(error), _layerSet(0), _logicalNotExpression(lne), 
     _logicalOrExpression(0)
{
}


C_primary_expression::C_primary_expression(
   C_layer_set *ls, SyntaxError * error)
   : C_production_grid(error), _layerSet(ls),  _logicalNotExpression(0), 
     _logicalOrExpression(0)
{
}


C_primary_expression* C_primary_expression::duplicate() const
{
   return new C_primary_expression(*this);
}


C_primary_expression::~C_primary_expression()
{
   delete _layerSet;
   delete _logicalNotExpression;
   delete _logicalOrExpression;
}

void C_primary_expression::checkChildren() 
{
   if (_layerSet) {
      _layerSet->checkChildren();
      if (_layerSet->isError()) {
         setError();
      }
   }
   if (_logicalNotExpression) {
      _logicalNotExpression->checkChildren();
      if (_logicalNotExpression->isError()) {
         setError();
      }
   }
   if (_logicalOrExpression) {
      _logicalOrExpression->checkChildren();
      if (_logicalOrExpression->isError()) {
         setError();
      }
   }
} 

void C_primary_expression::recursivePrint() 
{
   if (_layerSet) {
      _layerSet->recursivePrint();
   }
   if (_logicalNotExpression) {
      _logicalNotExpression->recursivePrint();
   }
   if (_logicalOrExpression) {
      _logicalOrExpression->recursivePrint();
   }
   printErrorMessage();
} 
