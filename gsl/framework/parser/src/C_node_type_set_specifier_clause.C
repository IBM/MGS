// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_node_type_set_specifier_clause.h"
#include "C_layer_set.h"
#include "C_logical_OR_expression.h"
#include "Grid.h"
#include "LensContext.h"
#include "GridLayerDescriptor.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_node_type_set_specifier_clause::internalExecute(LensContext *c)
{
   delete _storedContext;

   // Note: executing this copy constructor can lead to recursion in 
   // ConnectionScripFunctor, for now we'll set to NULL
   //_storedContext = new LensContext(c);
   _storedContext = 0;

}


const std::list<GridLayerDescriptor*>& 
C_node_type_set_specifier_clause::getLayers(Grid* g)
{
   if (g != _lastGrid) {
      if (_logicalOrExpression) {
         _logicalOrExpression->execute(_storedContext, g);
         _layers = _logicalOrExpression->getLayers();
      }
      else if (_layerSet) {
         _layerSet->execute(_storedContext, g);
         _layers = _layerSet->getLayers();
      }
      _lastGrid = g;
   }
   return _layers;
}


C_node_type_set_specifier_clause::C_node_type_set_specifier_clause(
   const C_node_type_set_specifier_clause& rv)
   : NodeTypeSet(rv), C_production(rv), _layerSet(0), _logicalOrExpression(0), 
     _storedContext(0), _lastGrid(rv._lastGrid)
{
   if (rv._logicalOrExpression) {
      _logicalOrExpression = rv._logicalOrExpression->duplicate();
   }
   if (rv._layerSet) {
      _layerSet = rv._layerSet->duplicate();
   }
   if (rv._storedContext) {
      _storedContext = new LensContext(*rv._storedContext);
   }
}


C_node_type_set_specifier_clause::C_node_type_set_specifier_clause(
   C_layer_set *l,  SyntaxError * error)
   : C_production(error), _layerSet(l), _logicalOrExpression(0), _storedContext(0), 
     _lastGrid(0)
{
}


C_node_type_set_specifier_clause::C_node_type_set_specifier_clause(
   C_logical_OR_expression *l, SyntaxError * error)
   : C_production(error), _layerSet(0), _logicalOrExpression(l), _storedContext(0), 
     _lastGrid(0)
{
}


C_node_type_set_specifier_clause* 
C_node_type_set_specifier_clause::duplicate() const
{
   return new C_node_type_set_specifier_clause(*this);
}


C_node_type_set_specifier_clause::~C_node_type_set_specifier_clause()
{
   delete _layerSet;
   delete _logicalOrExpression;
   delete _storedContext;
}

void C_node_type_set_specifier_clause::checkChildren() 
{
   if (_layerSet) {
      _layerSet->checkChildren();
      if (_layerSet->isError()) {
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

void C_node_type_set_specifier_clause::recursivePrint() 
{
   if (_layerSet) {
      _layerSet->recursivePrint();
   }
   if (_logicalOrExpression) {
      _logicalOrExpression->recursivePrint();
   }
   printErrorMessage();
} 
