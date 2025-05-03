// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_name_range_H
#define C_name_range_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_layer_name;
class Grid;
class GridLayerDescriptor;
class GslContext;
class SyntaxError;

class C_name_range : public C_production_grid
{
   public:
      C_name_range(const C_name_range&);
      C_name_range(C_layer_name*, C_layer_name *, SyntaxError *);
      virtual ~C_name_range();
      virtual C_name_range* duplicate() const;
      virtual void internalExecute(GslContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      C_layer_name* _fromLayerName;
      C_layer_name* _toLayerName;
      std::list<GridLayerDescriptor*> _layers;
};
#endif
