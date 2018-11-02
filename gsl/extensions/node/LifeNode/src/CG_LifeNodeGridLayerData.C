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
#include "CG_LifeNodeGridLayerData.h"
#include "CG_LifeNodeCompCategory.h"
#include "GridLayerDescriptor.h"
#include "LifeNode.h"
#include "NodeRelationalDataUnit.h"
#ifdef HAVE_MPI
#include "CG_LifeNodeProxy.h"
#endif
#include "Grid.h"
#include "GridLayerData.h"
#include "NodeInstanceAccessor.h"
#ifdef HAVE_MPI
#include "ShallowArray.h"
#endif
#include <numeric>

CG_LifeNodeGridLayerData::CG_LifeNodeGridLayerData(CG_LifeNodeCompCategory* compCategory, GridLayerDescriptor* gridLayerDescriptor, int gridLayerIndex) 
   : GridLayerData(compCategory, gridLayerDescriptor, gridLayerIndex)
{
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

      /* at the first layer then allocate all proxies, i.e. CG_LifeNodeProxy
       */
      if (sim->_proxy_count.count("LifeNode") == 0)
      {
         std::vector<int> proxy_from_ranks(sim->getNumProcesses(), 0);
         sim->_proxy_count["LifeNode"] = proxy_from_ranks;
         //std::map<Granule*, std::map<std::string, int>> _granulesFrom_NT_count;
         for (auto& kv : sim->_granulesFrom_NT_count)
         {
            int other_rank = kv.first->getPartitionId();
            if (other_rank != my_rank)
            {
               if (kv.second.count("LifeNode") > 0)
                  sim->_proxy_count["LifeNode"][other_rank] = kv.second["LifeNode"];
            }
         }

         //int total_nodes = std::accumulate(sim->_nodes_count["LifeNode"].begin(), sim->_nodes_count["LifeNode"].end(), 0); 
         //for (auto& kv : sim->_nodes_from_to_granules)
         //{
         //   /* test option 1 */
         //   //if (kv.first.first == "LifeNode")
         //   //{
         //   //   //std::vector<Granule*> unique_elements;
         //   //   //std::map<Granule*, int> granules_count;
         //   //   for (auto iter_v= kv.second.begin(); iter_v < kv.second.end(); iter_v++)
         //   //   {
         //   //      int other_rank = iter_v->first->getPartitionId();
         //   //      //auto iter = std::find(unique_elements.begin(), unique_elements.end(), iter_v->first);
         //   //      //auto iter = std::find(unique_elements.begin(), unique_elements.end(), iter_v->first);
         //   //      //if (other_rank != my_rank and iter == unique_elements.end())
         //   //      //if (other_rank != my_rank and (granules_count.count(iter_v->first) == 0 || 
         //   //      //         granules_couunt[iter_v->first] < )iter == unique_elements.end())
         //   //      if (other_rank != my_rank)
         //   //      {
         //   //         sim->_proxy_count["LifeNode"][other_rank] +=1;
         //   //         //unique_elements.push_back(iter_v->first);
         //   //      }
         //   //   }
         //   //   sim->_nodes_from_to_granules.erase(kv.first);
         //   //}

         //   ///* test option 2 */
         //   //for (auto iter_v= kv.second.begin(); iter_v < kv.second.end(); iter_v++)
         //   //{
         //   //   int to_rank = iter_v->second->getPartitionId();
         //   //   if (my_rank == to_rank)
         //   //   {
         //   //      int from_rank = iter_v->first->getPartitionId();
         //   //      if (from_rank != my_rank)
         //   //      {
         //   //         if (kv.first.first == "LifeNode")
         //   //         {
         //   //            sim->_proxy_count["LifeNode"][other_rank] +=1;
         //   //         }
         //   //      }
         //   //   }
         //   //}
         //   ////sim->_nodes_from_to_granules.erase(kv.first);
         //   
         //}

         //for (auto& kv : sim->_nodes_from_to_ND)
         //{
         //   std::vector<NodeDescriptor*> unique_elements;
         //   /* test option 3 */
         //   if (kv.first.first == "LifeNode")
         //   {
         //      //std::map<Granule*, int> granules_count;
         //      for (auto iter_v= kv.second.begin(); iter_v < kv.second.end(); iter_v++)
         //      {
         //         int other_rank = sim->getGranule(* iter_v->first)->getPartitionId();
         //         auto iter = std::find(unique_elements.begin(), unique_elements.end(), iter_v->first);
         //         if (other_rank != my_rank and iter == unique_elements.end())
         //         {
         //            sim->_proxy_count["LifeNode"][other_rank] +=1;
         //            unique_elements.push_back(iter_v->first);
         //         }
         //      }
         //      sim->_nodes_from_to_ND.erase(kv.first);
         //   }
         //}
         //compCategory->allocateProxies(sim->_proxy_count["LifeNode"][my_rank]);
         compCategory->allocateProxies(sim->_proxy_count["LifeNode"]);
      } 
      //if (sim->_proxy_count["LifeNode"])
      //{
      //   //compCategory->allocateProxies(sim->_proxy_count["LifeNode"][my_rank]);
      //   compCategory->allocateProxies(sim->_proxy_count["LifeNode"]);
      //   sim->_proxy_count.erase("LifeNode");
      //} 
   }
#endif
#if defined(HAVE_GPU) && defined(__NVCC__)
   if (! sim->isGranuleMapperPass()) {
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
#if defined(HAVE_GPU) && defined(__NVCC__)
   }
#endif
}

NodeInstanceAccessor* CG_LifeNodeGridLayerData::getNodeInstanceAccessors() 
{
   return _nodeInstanceAccessors;
}

CG_LifeNodeGridLayerData::~CG_LifeNodeGridLayerData() 
{
   delete[] _nodeInstanceAccessors;
}

