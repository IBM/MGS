// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_layer_set_H
#define C_layer_set_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_layer_entry;
class LensContext;
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
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      std::list<C_layer_entry*>* _listLayerEntry;
      std::list<GridLayerDescriptor*> _layers;
      Grid* _lastGrid;
};
#endif
