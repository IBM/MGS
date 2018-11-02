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

#include "NoConnectConnector.h"
#include "Grid.h"
#include "EdgeType.h"
#include "Node.h"
#include "NodeDescriptor.h"
#include "ParameterSet.h"
#include "Simulation.h"
#include "Repertoire.h"
#include "Edge.h"
#include "GridLayerDescriptor.h"
#include "Simulation.h"
#include "Constant.h"
#include "Variable.h"
#include "VariableDescriptor.h"
#include "NDPairList.h"
#include "NodeSet.h"
#include "EdgeSet.h"
#include "NodeCompCategoryBase.h"          // added by Jizhu Lu on 02/28/2006
#include "GridLayerData.h"                 // added by Jizhu Lu on 02/28/2006
#include <iostream>
#include <cassert>

#include <list>
#include <utility>

NoConnectConnector::NoConnectConnector()
{
}

NoConnectConnector::~NoConnectConnector()
{
}

void NoConnectConnector::nodeToNode(
   NodeDescriptor *from, ParameterSet *outAttrPSet, NodeDescriptor *to, 
   ParameterSet *inAttrPSet, Simulation* sim)
{
#if defined(HAVE_GPU) && defined(__NVCC__)
   // store the information about 'from' and 'to' partitionId
   // NOTE: The nodedescriptor can represent 2 different nodetypes, e.g. LifeNode and PredetorNode
#ifdef HAVE_MPI
   int myRank = sim->getRank();
   //int fromPartitionId = sim->getGranule(*from)->getPartitionId();
   //int toPartitionId = sim->getGranule(*to)->getPartitionId();
   auto nodetype_from = from->getGridLayerData()->getNodeCompCategoryBase()->getModelName();
   auto nodetype_to = to->getGridLayerData()->getNodeCompCategoryBase()->getModelName();
   auto key = std::make_pair(nodetype_from, nodetype_to);
   //if (sim->_nodes_from_to_granules.count(key) == 0)
   {
         //std::vector<int> nodes_on_ranks(sim->getNumProcesses(), 0);
         //std::vector<int> all_ranks(sim->getNumProcesses(), nodes_on_ranks);
      auto value = std::make_pair(sim->getGranule(*from), sim->getGranule(*to));
      sim->_nodes_from_to_granules[key].push_back(value);
   }
   {
      if (sim->_granulesFrom_NT_count.count(sim->getGranule(*from)) == 0)
      {
         std::map<std::string, int> tmp { { nodetype_from, 1 } };
         //tmp[nodetype_from] = 1;
         sim->_granulesFrom_NT_count[sim->getGranule(*from)] = tmp;
         if (sim->_nodes_ND.find(from) == sim->_nodes_ND.end())
               sim->_nodes_ND[from] = 1;
      }
      else{
         if (sim->_granulesFrom_NT_count[sim->getGranule(*from)].count(nodetype_from) == 0)
            sim->_granulesFrom_NT_count[sim->getGranule(*from)][nodetype_from] = 1;
         else
         {
            if (sim->_nodes_ND.find(from) == sim->_nodes_ND.end())
            {
               sim->_nodes_ND[from] = 1;
               sim->_granulesFrom_NT_count[sim->getGranule(*from)][nodetype_from] += 1;
            }
         }
      }
   }
   //{
   //   if (sim->_nodes_ND.count(from) == 0)
   //   {
   //      sim->_nodes_ND[from] = sim->_nodes_ND.size();
   //   }
   //   if (sim->_nodes_ND.count(to) == 0)
   //   {
   //      sim->_nodes_ND[to] = sim->_nodes_ND.size();
   //   }
   //   if (sim->_granulesFrom_and_NDIdentifier.count(sim->getGranule(*from)) == 0)
   //   {
   //      sim->_granules_and_NDIdentifier[sim->getGranule(*from)] = std::vector<int>();
   //   }
   //   if (sim->_granules_and_NDIdentifier.count(sim->getGranule(*from)) == 0)
   //   {
   //      sim->_granules_and_NDIdentifier[sim->getGranule(*from)] = std::vector<int>();
   //   }
   //   else{
   //      sim->_granules_and_NDIdentifier[sim->getGranule(*from)].push_back(sim->_nodes_ND[from]); 
   //   }
   //   if (sim->_granules_and_NDIdentifier.count(sim->getGranule(*to)) == 0)
   //   {
   //      sim->_granules_and_NDIdentifier[sim->getGranule(*to)] = std::vector<int>();
   //   }
   //   else{
   //      sim->_granules_and_NDIdentifier[sim->getGranule(*to)].push_back(sim->_nodes_ND[to]); 
   //   }
   //   //auto value = std::make_pair(from, to);
   //   //if (sim->_current_ND_index.cout(nodetype_from) == 0)
   //   //   sim->_current_ND_index[nodetype_from] = 0;
   //   //else
   //   {
   //      if (sim->_nodes_ND.count(from) == 0)
   //      {
   //         sim->_nodes_ND[from] = 0;
   //      }
   //      else{
   //         sim->_nodes_ND[from] += 1;
   //      }
   //   }
   //   //if (sim->_current_ND_index.cout(nodetype_to) == 0)
   //   //   sim->_current_ND_index[nodetype_to] = 0;
   //   //else
   //   {
   //      if (sim->_nodes_ND.count(to) == 0)
   //      {
   //         sim->_nodes_ND[to] = 0;
   //      }
   //      else{
   //         sim->_nodes_ND[to] += 1;
   //      }
   //   }

   //   auto value = std::make_pair(sim->_nodes_ND[from], sim->_nodes_ND[to]);

   //   sim->_nodes_from_to_ND[key].push_back(value);
   //}
   //if (sim->_proxy_count.count(nodetype_from) == 0)
   //{
   //   std::vector<int> proxy_from_ranks(sim->getNumProcesses(), 0);
   //   sim->_proxy_count[nodetype_from] = proxy_from_ranks;
   //   //for (int i = 0; i < sim->getNumProcesses())
   //   //{
   //   //   sim->_proxy_count[nodetype_to] = proxy_from_ranks;
   //   //}
   //}


   //if ( (fromPartitionId != myRank) && (toPartitionId != myRank) ){
   //   /* both nodes are not in current memory space: do nothing */
   //}
   //else if ( (fromPartitionId != myRank) && (toPartitionId == myRank) ) {
   //   /* "from" is not in current memory space but "to" is: 
   //      check for "from" proxy, allocate if necessary, connect */
   //   if (from->getNode()==0) {
   //      from->getGridLayerData()->getNodeCompCategoryBase()->allocateProxy(fromPartitionId, from);
   //      assert(from->getNode());
   //   }
   //   //from->getNode()->addPostNode(to, outAttrPSet); // not added yet because use not clear but cost is
   //   to->getNode()->addPreNode(from, inAttrPSet);
   //}
   //else if (fromPartitionId == myRank && toPartitionId != myRank) {
   //   /* "from" is in current memory space but "to" is not: 
   //      add the "from" node to the node list which will 
   //      send message to the memory space "to" is in */
   //   from->getGridLayerData()->getNodeCompCategoryBase()->addToSendMap(toPartitionId, from->getNode());
   //   //from->getNode()->addPostNode(to, outAttrPSet);
   //}
   //else
   {

#endif
      /* Both "from" and "to" are in current memory space: connect as usual */
      /* For the first pass, Computation cost including Computation Time, Memory, 
         Communication Overhead of the two nodes need to be added together as
         the total cost of this connection.  -- Jizhu Lu  01/24/06 */

   //   from->getNode()->addPostNode(to, outAttrPSet);
   //   to->getNode();
   //   to->getNode()->addPreNode(from, inAttrPSet);

#ifdef  HAVE_MPI
   }
#endif

#endif
}

void NoConnectConnector::nodeToNodeWithEdge(
   EdgeType *et, ParameterSet *edgeInit, NodeDescriptor *from, 
   ParameterSet *outAttrPSet, NodeDescriptor *to, ParameterSet *inAttrPSet, Simulation* sim)
{
}

void NoConnectConnector::variableToNodeSet(
   VariableDescriptor* source, NodeSet* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   std::vector<NodeDescriptor*> nodes;
   destination->getNodes(nodes);
   if (nodes.size()>0) {
     std::vector<NodeDescriptor*>::iterator it = nodes.begin(), end=nodes.end();
     Granule *toGran, 
       *fromGran=sim->getGranule(*source);
     std::map<Granule*, int> granHist;
     std::map<Granule*, int>::iterator miter;
     int max=0;
     for (; it!=end; ++it) {
       toGran = sim->getGranule(**it);
       miter=granHist.find(toGran);
       if (miter==granHist.end()) granHist[toGran]=0;
       int count=++granHist[toGran];
       if (count>max) max=count;
     }
     assert (granHist.size()>0);
     for (miter=granHist.begin(); miter!=granHist.end() && miter->second!=max; ++miter) {}
     fromGran->setDepends(miter->first);
   }
}

void NoConnectConnector::variableToEdgeSet(
   VariableDescriptor* source, EdgeSet* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
}

void NoConnectConnector::variableToVariable(
   VariableDescriptor* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
}

void NoConnectConnector::nodeSetToVariable(
   NodeSet* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   std::vector<NodeDescriptor*> nodes;
   source->getNodes(nodes);
   if (nodes.size()>0) {
     std::vector<NodeDescriptor*>::iterator it = nodes.begin(), end=nodes.end();
     Granule *fromGran, 
       *toGran=sim->getGranule(*destination);
     std::map<Granule*, int> granHist;
     std::map<Granule*, int>::iterator miter;
     int max=0;
     for (; it!=end; ++it) {
       fromGran = sim->getGranule(**it);
       miter=granHist.find(fromGran);
       if (miter==granHist.end()) granHist[fromGran]=0;
       int count=++granHist[fromGran];
       if (count>max) max=count;
     }
     for (miter=granHist.begin(); miter!=granHist.end() && miter->second!=max; ++miter) {}
     toGran->setDepends(miter->first);
   }
}

void NoConnectConnector::edgeSetToVariable(
   EdgeSet* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
}
