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

#ifndef C_name_range_H
#define C_name_range_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_layer_name;
class Grid;
class GridLayerDescriptor;
class LensContext;
class SyntaxError;

class C_name_range : public C_production_grid
{
   public:
      C_name_range(const C_name_range&);
      C_name_range(C_layer_name*, C_layer_name *, SyntaxError *);
      virtual ~C_name_range();
      virtual C_name_range* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      C_layer_name* _fromLayerName;
      C_layer_name* _toLayerName;
      std::list<GridLayerDescriptor*> _layers;
};
#endif
