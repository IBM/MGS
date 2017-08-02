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

#include "C_layer_set.h"
#include "C_layer_entry.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "SyntaxError.h"
#include "C_production_grid.h"

void C_layer_set::internalExecute(LensContext *c, Grid* g)
{
   if (g != _lastGrid) {
      _layers.clear();
      _lastGrid = g;
   }
   else _layers.sort();
   std::list<C_layer_entry*>::iterator entry, end=_listLayerEntry->end();
   for (entry = _listLayerEntry->begin(); entry != end; ++entry) {
      C_layer_entry* e = (*entry);
      e->execute(c, g);
      std::list<GridLayerDescriptor*> lin = e->getLayers();
      lin.sort();
      _layers.merge(lin);
      _layers.unique();
   }
}


C_layer_set::C_layer_set(C_layer_entry* le, SyntaxError* error)
   : C_production_grid(error), _listLayerEntry(0), _lastGrid(0)
{
   _listLayerEntry = new std::list<C_layer_entry*>;
   _listLayerEntry->push_back(le);
}


C_layer_set::C_layer_set(C_layer_set *ls, C_layer_entry *le, 
			 SyntaxError * error)
   : C_production_grid(error), _listLayerEntry(0), _lastGrid(0)
{
   _listLayerEntry = ls->releaseSet();
   _listLayerEntry->push_back(le);
   delete ls;
}


C_layer_set::C_layer_set(const C_layer_set& rv)
   : C_production_grid(rv), _listLayerEntry(0), _layers(rv._layers), 
     _lastGrid(rv._lastGrid)
{
   _listLayerEntry = new std::list<C_layer_entry*>;
   if (rv._listLayerEntry) {
      std::list<C_layer_entry*>::iterator iter, 
	 end = rv._listLayerEntry->end();
      for (iter = rv._listLayerEntry->begin(); iter != end; ++iter) {
	 _listLayerEntry->push_back((*iter)->duplicate());
      }
   }
}


std::list<C_layer_entry*>* C_layer_set::releaseSet()
{
   std::list<C_layer_entry*> *retval = _listLayerEntry;
   _listLayerEntry = 0;
   return retval;
}


const std::list<GridLayerDescriptor*>& C_layer_set::getLayers() const
{
   return _layers;
}


C_layer_set* C_layer_set::duplicate() const
{
   return new C_layer_set(*this);
}


C_layer_set::~C_layer_set()
{
   if (_listLayerEntry) {
      std::list<C_layer_entry*>::iterator i, end =_listLayerEntry->end();
      for (i = _listLayerEntry->begin(); i != end; ++i) {
	 delete *i;
      }
      delete _listLayerEntry;
   }
}

void C_layer_set::checkChildren() 
{
} 

void C_layer_set::recursivePrint() 
{
   printErrorMessage();
} 
