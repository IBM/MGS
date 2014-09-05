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

#ifndef C_layer_entry_H
#define C_layer_entry_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_name_range;
class C_layer_name;
class LensContext;
class Grid;
class GridLayerDescriptor;
class SyntaxError;

class C_layer_entry : public C_production_grid
{
   public:
      C_layer_entry(const C_layer_entry&);
      C_layer_entry(C_name_range *, SyntaxError *);
      C_layer_entry(C_layer_name *, SyntaxError *);
      virtual ~C_layer_entry ();
      virtual C_layer_entry* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      C_name_range* _nameRange;
      C_layer_name* _layerName;
      std::list<GridLayerDescriptor*> _layers;
};
#endif
