// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_name_range.h"
#include "C_layer_name.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "GslContext.h"
#include "C_production_grid.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_name_range::internalExecute(GslContext *c, Grid* g)
{

   _fromLayerName->execute(c);
   _toLayerName->execute(c);

   _layers.clear();
   const std::vector<GridLayerDescriptor*>& glayers = g->getLayers();
   std::vector<GridLayerDescriptor*>::const_iterator iter, end = glayers.end();

   bool toggle = false;
   for (iter = glayers.begin(); iter != end; ++iter) {
      GridLayerDescriptor* gld = *iter;
      std::string gldName = gld->getName();
      if (gldName == _fromLayerName->getName())
         toggle = true;
      if (toggle) _layers.push_back(gld);
      if (gldName == _toLayerName->getName())
         toggle = false;
   }

   // check if begin or end was not found
   if (toggle || _layers.size() == 0) {
      std::string mes = "invalid layer name range specified";
      throwError(mes);
   }
   _layers.sort();
   _layers.unique();
}


const std::list<GridLayerDescriptor*>& C_name_range::getLayers() const
{
   return _layers;
}


C_name_range::C_name_range(C_layer_name* from, C_layer_name* to, 
			   SyntaxError* error)
   : C_production_grid(error), _fromLayerName(from), _toLayerName(to)
{
}


C_name_range::C_name_range(const C_name_range& rv)
   : C_production_grid(rv), _fromLayerName(0), _toLayerName(0), 
     _layers(rv._layers)
{
   if (rv._fromLayerName) {
      _fromLayerName = rv._fromLayerName->duplicate();
   }
   if (rv._toLayerName) {
      _toLayerName = rv._toLayerName->duplicate();
   }
}


C_name_range* C_name_range::duplicate() const
{
   return new C_name_range(*this);
}


C_name_range::~C_name_range()
{
   delete _fromLayerName;
   delete _toLayerName;
}

void C_name_range::checkChildren() 
{
   if (_fromLayerName) {
      _fromLayerName->checkChildren();
      if (_fromLayerName->isError()) {
         setError();
      }
   }
   if (_toLayerName) {
      _toLayerName->checkChildren();
      if (_toLayerName->isError()) {
         setError();
      }
   }
} 

void C_name_range::recursivePrint() 
{
   if (_fromLayerName) {
      _fromLayerName->recursivePrint();
   }
   if (_toLayerName) {
      _toLayerName->recursivePrint();
   }
   printErrorMessage();
} 
