// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CG_LifeNodeNodeAccessor.h"
#include "SyntaxErrorException.h"
#include "CG_LifeNodeGridLayerData.h"
#include "GridLayerDescriptor.h"
#include "Node.h"
#include "Simulation.h"
#include "LifeNode.h"
#include "NodeAccessor.h"
#include <memory>
#include <string>
#include <vector>

int CG_LifeNodeNodeAccessor::getNbrUnits() 
{
   return _gridLayerData->getNbrUnits();
}

GridLayerDescriptor* CG_LifeNodeNodeAccessor::getGridLayerDescriptor() 
{
   return _gridLayerDescriptor;
}

std::string CG_LifeNodeNodeAccessor::getModelName() 
{
   return _gridLayerDescriptor->getModelName();
}

NodeDescriptor* CG_LifeNodeNodeAccessor::getNodeDescriptor(const std::vector<int>& coords, int densityIndex) 
{
   return getNodeDescriptor(_gridLayerDescriptor->getGrid()->getNodeIndex(coords), densityIndex);
}

NodeDescriptor* CG_LifeNodeNodeAccessor::getNodeDescriptor(int nodeIndex, int densityIndex) 
{
   int density = _gridLayerDescriptor->getDensity(nodeIndex);
   if (densityIndex >= density) {
      std::cerr << " CG_LifeNodeNodeAccessor " << DENSITY_ERROR_MESSAGE << std::endl;
      throw SyntaxErrorException(DENSITY_ERROR_MESSAGE);
   }
   if (_gridLayerDescriptor->isUniform()) {
      // idx can be computed by simply multiplying by density
      nodeIndex *= density;
   } else {
      // density must be non-uniform, use nodeOffsets set by CompCategory
      if (_gridLayerData->getNodeOffsets().size()) {
         nodeIndex = _gridLayerData->getNodeOffsets()[nodeIndex];
      } else {
         std::cerr << OFFSET_ERROR_MESSAGE << std::endl;
         throw SyntaxErrorException(OFFSET_ERROR_MESSAGE);
      }
   }
   nodeIndex += densityIndex;
   return (_gridLayerData->getNodeInstanceAccessors()) + nodeIndex;
}

CG_LifeNodeNodeAccessor::CG_LifeNodeNodeAccessor(Simulation& sim, GridLayerDescriptor* gridLayerDescriptor, CG_LifeNodeGridLayerData* gridLayerData) 
   : NodeAccessor(), _sim(sim), _gridLayerDescriptor(gridLayerDescriptor), _gridLayerData(gridLayerData)
{
}

CG_LifeNodeNodeAccessor::~CG_LifeNodeNodeAccessor() 
{
}

void CG_LifeNodeNodeAccessor::duplicate(std::unique_ptr<CG_LifeNodeNodeAccessor>& dup) const
{
   dup.reset(new CG_LifeNodeNodeAccessor(*this));
}

void CG_LifeNodeNodeAccessor::duplicate(std::unique_ptr<NodeAccessor>& dup) const
{
   dup.reset(new CG_LifeNodeNodeAccessor(*this));
}

