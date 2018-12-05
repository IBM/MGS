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

#ifndef CG_LifeNodeNodeAccessor_H
#define CG_LifeNodeNodeAccessor_H

#include "Lens.h"
#include "LifeNode.h"
#include "NodeAccessor.h"
#include <memory>
#include <string>
#include <vector>

class CG_LifeNodeGridLayerData;
class GridLayerDescriptor;
class Node;
class Simulation;

class CG_LifeNodeNodeAccessor : public NodeAccessor
{
   public:
      virtual int getNbrUnits();
      virtual GridLayerDescriptor* getGridLayerDescriptor();
      virtual std::string getModelName();
      virtual NodeDescriptor* getNodeDescriptor(const std::vector<int>& coords, int densityIndex);
      virtual NodeDescriptor* getNodeDescriptor(int nodeIndex, int densityIndex);
      CG_LifeNodeNodeAccessor(Simulation& sim, GridLayerDescriptor* gridLayerDescriptor, CG_LifeNodeGridLayerData* gridLayerData);
      virtual ~CG_LifeNodeNodeAccessor();
      virtual void duplicate(std::unique_ptr<CG_LifeNodeNodeAccessor>& dup) const;
      virtual void duplicate(std::unique_ptr<NodeAccessor>& dup) const;
   private:
      Simulation& _sim;
      GridLayerDescriptor* _gridLayerDescriptor;
      CG_LifeNodeGridLayerData* _gridLayerData;
};

#endif
