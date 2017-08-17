// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
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
