// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_layer_set_H
#define C_layer_set_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_layer_entry;
class GslContext;
class Grid;
class GridLayerDescriptor;
class SyntaxError;

class C_layer_set : public C_production_grid
{
   public:
      C_layer_set(const C_layer_set&);
      C_layer_set(C_layer_entry *, SyntaxError *);
      C_layer_set(C_layer_set*, C_layer_entry *, SyntaxError *);
      std::list<C_layer_entry*>* releaseSet();
      virtual ~C_layer_set();
      virtual C_layer_set* duplicate() const;
      virtual void internalExecute(GslContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      std::list<C_layer_entry*>* _listLayerEntry;
      std::list<GridLayerDescriptor*> _layers;
      Grid* _lastGrid;
};
#endif
