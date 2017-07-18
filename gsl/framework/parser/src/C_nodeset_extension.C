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

#include "C_nodeset_extension.h"
#include "C_node_type_set_specifier.h"
#include "C_index_set_specifier.h"
#include "LensContext.h"
#include "Grid.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_nodeset_extension::internalExecute(LensContext *c)
{
   if(_indexSetSpecifier) {
      _indexSetSpecifier->execute(c);
      _indices = _indexSetSpecifier->getIndices();
   }
   delete _storedContext;

   // Note: executing this copy constructor can lead to recursion in 
   // ConnectionScriptFunctor, for now we'll set to NULL
   //_storedContext = new LensContext(c);
   _storedContext = 0;
}

const std::list<GridLayerDescriptor*>& C_nodeset_extension::getLayers(Grid* g)
{
   if (g != _lastGrid || _layers == 0) {
      delete _layers;
      if(_nodeTypeSetSpecifier) {
         _nodeTypeSetSpecifier->execute(_storedContext, g);
         _layers = new std::list<GridLayerDescriptor*>(
	    _nodeTypeSetSpecifier->getLayers() );
      }
      else {
         std::vector<GridLayerDescriptor*> const & glayers = g->getLayers();
         std::vector<GridLayerDescriptor*>::const_iterator iter, 
	    end = glayers.end();
         _layers = new std::list<GridLayerDescriptor*>();
         for (iter = glayers.begin(); iter != end; ++iter) {
	    _layers->push_back(*iter);
	 }
      }
      _lastGrid = g;
   }
   return *_layers;
}

const std::vector<int>& C_nodeset_extension::getIndices()
{
   return _indices;
}

C_nodeset_extension::C_nodeset_extension(const C_nodeset_extension& rv)
   : C_production(rv), _nodeTypeSetSpecifier(0), _indexSetSpecifier(0),
     _storedContext(0), _layers(0), _lastGrid(rv._lastGrid)
{
   if(rv._nodeTypeSetSpecifier) {
      _nodeTypeSetSpecifier = rv._nodeTypeSetSpecifier->duplicate();
   }
   if(rv._indexSetSpecifier) {
      _indexSetSpecifier = rv._indexSetSpecifier->duplicate();
   }
   if(rv._storedContext) {
      _storedContext = new LensContext(*rv._storedContext);
   }
   if(rv._layers) {
      _layers = new std::list<GridLayerDescriptor*>(*(rv._layers));
   }
}

C_nodeset_extension::C_nodeset_extension(
   C_node_type_set_specifier *n, C_index_set_specifier *i, SyntaxError * error)
   : C_production(error), _nodeTypeSetSpecifier(n), _indexSetSpecifier(i), 
     _storedContext(0), _layers(0), _lastGrid(0)
{
}

C_nodeset_extension::C_nodeset_extension(
   C_index_set_specifier *i, SyntaxError * error)
   : C_production(error), _nodeTypeSetSpecifier(0), _indexSetSpecifier(i), 
     _storedContext(0), _layers(0), _lastGrid(0)
{
}

C_nodeset_extension::C_nodeset_extension(
   C_node_type_set_specifier *n, SyntaxError * error)
   : C_production(error), _nodeTypeSetSpecifier(n), _indexSetSpecifier(0), 
     _storedContext(0),_layers(0), _lastGrid(0)
{
}

C_nodeset_extension* C_nodeset_extension::duplicate() const
{
   return new C_nodeset_extension(*this);
}


C_nodeset_extension::~C_nodeset_extension()
{
   delete _nodeTypeSetSpecifier;
   delete _indexSetSpecifier;
   delete _storedContext;
   delete _layers;
}

void C_nodeset_extension::checkChildren() 
{
   if (_nodeTypeSetSpecifier) {
      _nodeTypeSetSpecifier->checkChildren();
      if (_nodeTypeSetSpecifier->isError()) {
         setError();
      }
   }
   if (_indexSetSpecifier) {
      _indexSetSpecifier->checkChildren();
      if (_indexSetSpecifier->isError()) {
         setError();
      }
   }
} 

void C_nodeset_extension::recursivePrint() 
{
   if (_nodeTypeSetSpecifier) {
      _nodeTypeSetSpecifier->recursivePrint();
   }
   if (_indexSetSpecifier) {
      _indexSetSpecifier->recursivePrint();
   }
   printErrorMessage();
} 
