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
