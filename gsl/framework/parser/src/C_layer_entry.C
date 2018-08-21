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

#include "C_layer_entry.h"
#include "C_name_range.h"
#include "C_layer_name.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "LensContext.h"
#include "Repertoire.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production_grid.h"


void C_layer_entry::internalExecute(LensContext *c, Grid* g)
{
   _layers.clear();
   if (_nameRange) {
      _nameRange->execute(c, g);
      _layers = _nameRange->getLayers();
   }

   else if (_layerName) {
      _layerName ->execute(c);

      const std::vector<GridLayerDescriptor*>& glayers = g->getLayers();
      std::string name = _layerName->getName();
      std::vector<GridLayerDescriptor*>::const_iterator iter, 
	 end = glayers.end();

      bool found = false;
      for (iter = glayers.begin(); iter != end; ++iter) {
         GridLayerDescriptor* gld = (*iter);
         if (gld->getName() == name) {
            _layers.push_back(gld);
            found = true;
            break;
         }
      }

      if (!found) {
	 std::string mes = "invalid layer name specified";
	 throwError(mes);
      }
   }
   _layers.sort();
   _layers.unique();
}


C_layer_entry::C_layer_entry(C_layer_name *n, SyntaxError * error)
   : C_production_grid(error), _nameRange(0), _layerName(n)
{
}


C_layer_entry::C_layer_entry(C_name_range *nr, SyntaxError * error)
   : C_production_grid(error), _nameRange(nr), _layerName(0)
{
}


C_layer_entry::C_layer_entry(const C_layer_entry& rv)
   : C_production_grid(rv), _nameRange(0), _layerName(0), _layers(rv._layers)
{
   if (rv._nameRange) {
      _nameRange = rv._nameRange->duplicate();
   }
   if (rv._layerName) {
      _layerName = rv._layerName->duplicate();
   }

}


const std::list<GridLayerDescriptor*>& C_layer_entry::getLayers() const
{
   return _layers;
}


C_layer_entry* C_layer_entry::duplicate() const
{
   return new C_layer_entry(*this);
}


C_layer_entry::~C_layer_entry()
{
   delete _nameRange;
   delete _layerName;
}

void C_layer_entry::checkChildren() 
{
   if (_nameRange) {
      _nameRange->checkChildren();
      if (_nameRange->isError()) {
         setError();
      }
   }
   if (_layerName) {
      _layerName->checkChildren();
      if (_layerName->isError()) {
         setError();
      }
   }
} 

void C_layer_entry::recursivePrint() 
{
   if (_nameRange) {
      _nameRange->recursivePrint();
   }
   if (_layerName) {
      _layerName->recursivePrint();
   }
   printErrorMessage();
} 
