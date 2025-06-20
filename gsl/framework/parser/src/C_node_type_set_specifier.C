// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_node_type_set_specifier.h"
#include "C_node_type_set_specifier_clause.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "SyntaxError.h"
#include "C_production_grid.h"

void C_node_type_set_specifier::internalExecute(GslContext *c, Grid* g)
{
   _nodeTypeSetSpecifierClause->execute(c);
   _layers = _nodeTypeSetSpecifierClause->getLayers(g);
}


const std::list<GridLayerDescriptor*>& 
C_node_type_set_specifier::getLayers() const
{
   return _layers;
}


C_node_type_set_specifier::C_node_type_set_specifier(
   const C_node_type_set_specifier& rv)
   : C_production_grid(rv), _nodeTypeSetSpecifierClause(0), _layers(rv._layers)
{
   if (rv._nodeTypeSetSpecifierClause) {
      _nodeTypeSetSpecifierClause = 
	 rv._nodeTypeSetSpecifierClause->duplicate();
   }
   _layers = rv._layers;
}


C_node_type_set_specifier::C_node_type_set_specifier(
   C_node_type_set_specifier_clause *n, SyntaxError * error)
   : C_production_grid(error), _nodeTypeSetSpecifierClause(n), _layers(0)
{
}


C_node_type_set_specifier* C_node_type_set_specifier::duplicate() const
{
   return new C_node_type_set_specifier(*this);
}


C_node_type_set_specifier::~C_node_type_set_specifier()
{
   delete _nodeTypeSetSpecifierClause;
}

void C_node_type_set_specifier::checkChildren() 
{
   if (_nodeTypeSetSpecifierClause) {
      _nodeTypeSetSpecifierClause->checkChildren();
      if (_nodeTypeSetSpecifierClause->isError()) {
         setError();
      }
   }
} 

void C_node_type_set_specifier::recursivePrint() 
{
   if (_nodeTypeSetSpecifierClause) {
      _nodeTypeSetSpecifierClause->recursivePrint();
   }
   printErrorMessage();
} 
