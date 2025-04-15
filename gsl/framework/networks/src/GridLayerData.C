// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GridLayerData.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "NodeCompCategoryBase.h"
#include "Simulation.h"

GridLayerData::GridLayerData(NodeCompCategoryBase* compCategory, 
			     GridLayerDescriptor* gridLayerDescriptor,
			     int gridLayerIndex)
   : _nbrUnits(0), _nbrNodesAllocated(0), _gridLayerDescriptor(gridLayerDescriptor), 
     _compCategory(compCategory), _gridLayerIndex(gridLayerIndex)
{     
  int gridNodes = _gridLayerDescriptor->getGrid()->getNbrGridNodes();
  if (_gridLayerDescriptor->isUniform()) {
    _nbrUnits = gridNodes * _gridLayerDescriptor->isUniform();
  } else {
    // grid has non-uniform density; must count points
    // and construct nodeOffsets for NodeAccessor
    _nbrUnits=0;
    for (int i = 0; i < gridNodes; ++i) {
      _nodeOffsets.push_back(_nbrUnits);
      _nbrUnits += _gridLayerDescriptor->getDensity(i);
    }
  }
}

GridLayerData::~GridLayerData()
{
}
