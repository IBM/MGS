// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-12-03-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CG_LifeNodeGridLayerData.h"
#include "CG_LifeNodeCompCategory.h"
#include "GridLayerDescriptor.h"
#include "LifeNode.h"
#include "NodeRelationalDataUnit.h"
#if defined(HAVE_MPI)
#include "CG_LifeNodeProxy.h"
#endif
#include "Grid.h"
#include "GridLayerData.h"
#include "NodeInstanceAccessor.h"
#if defined(HAVE_MPI)
#include "ShallowArray.h"
#endif

CG_LifeNodeGridLayerData::CG_LifeNodeGridLayerData(CG_LifeNodeCompCategory* compCategory, GridLayerDescriptor* gridLayerDescriptor, int gridLayerIndex) 
   : GridLayerData(compCategory, gridLayerDescriptor, gridLayerIndex){
   _nodeInstanceAccessors = new NodeInstanceAccessor[_nbrUnits];
   // set gridNode index for each node's relational information
   int top;
   int uniformDensity = _gridLayerDescriptor->isUniform();
   int gridNodes = _gridLayerDescriptor->getGrid()->getNbrGridNodes();
   Simulation *sim = &compCategory->getSimulation();
   unsigned my_rank = sim->getRank();
#if defined(HAVE_GPU) && defined(__NVCC__)
   if (sim->isGranuleMapperPass()) {
      if (sim->_nodes_count.count("LifeNode") == 0)
      { 
         std::vector<int> nodes_on_ranks(sim->getNumProcesses(), 0);
         sim->_nodes_count["LifeNode"] = nodes_on_ranks;
         std::vector<Granule*> nodes_on_partitions;
         sim->_nodes_granules["LifeNode"] = nodes_on_partitions;
      }
      for(int n = 0, gn = 0; gn < gridNodes; ++gn) {
         if (uniformDensity) {
            top = (gn + 1) * uniformDensity;
         } else {
               top = _nodeOffsets[gn] + _gridLayerDescriptor->getDensity(gn);
         }
         for (; n < top; ++n) {
            _nodeInstanceAccessors[n].setNodeIndex(gn);
            _nodeInstanceAccessors[n].setIndex(n);
            _nodeInstanceAccessors[n].setGridLayerData(this);
            /* it means instance 'LifeNode' at index 'i'
             * is created on partition 'sim->_nodes_granules["LifeNode"][i]->getPartitionId()'
             */
            sim->_nodes_granules["LifeNode"].push_back(sim->getGranule(_nodeInstanceAccessors[n]));
         }
      }
   }
   if (sim->isSimulatePass())
   {
      /* at the first layer of 'LifeNode' then allocate memory */
      if (sim->_nodes_count["LifeNode"][my_rank] == 0)
      {
         for(int n = 0; n < sim->_nodes_granules["LifeNode"].size(); ++n) {
            if (sim->_nodes_granules["LifeNode"][n]->getPartitionId()  == my_rank)
            {
               sim->_nodes_count["LifeNode"][my_rank] += 1;
            }
         }
         sim->_nodes_granules.erase("LifeNode");
         compCategory->allocateNodes(sim->_nodes_count["LifeNode"][my_rank]);
      }
   }
#endif
   for(int n = 0, gn = 0; gn < gridNodes; ++gn) {
      if (uniformDensity) {
         top = (gn + 1) * uniformDensity;
      } else {
         top = _nodeOffsets[gn] + _gridLayerDescriptor->getDensity(gn);
      }
      for (; n < top; ++n) {
         _nodeInstanceAccessors[n].setNodeIndex(gn);
         _nodeInstanceAccessors[n].setIndex(n);
         _nodeInstanceAccessors[n].setGridLayerData(this);

         if (sim->isSimulatePass() && sim->getGranule(_nodeInstanceAccessors[n])->getPartitionId() == my_rank) {
            compCategory->allocateNode(&_nodeInstanceAccessors[n]);
         }
      }
   }
}

NodeInstanceAccessor* CG_LifeNodeGridLayerData::getNodeInstanceAccessors() 
{
   return _nodeInstanceAccessors;
}

CG_LifeNodeGridLayerData::~CG_LifeNodeGridLayerData() 
{
   delete[] _nodeInstanceAccessors;
}

