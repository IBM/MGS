// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GranuleConnector.h"
#include "Grid.h"
#include "EdgeType.h"
#include "Node.h"
#include "NodeDescriptor.h"
#include "ParameterSet.h"
#include "Simulation.h"
#include "Repertoire.h"
#include "Edge.h"
#include "GridLayerDescriptor.h"
#include "Granule.h"
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

GranuleConnector::GranuleConnector()
{
}

GranuleConnector::~GranuleConnector()
{
}

void GranuleConnector::nodeToNode(
   NodeDescriptor *from, ParameterSet *outAttrPSet, NodeDescriptor *to, 
   ParameterSet *inAttrPSet, Simulation* sim)
{
  Granule* fromGran = sim->getGranule(*from);
  Granule* toGran = sim->getGranule(*to);

  fromGran->addConnection(toGran);
}

void GranuleConnector::nodeToNodeWithEdge(
   EdgeType *et, ParameterSet *edgeInit, NodeDescriptor *from, 
   ParameterSet *outAttrPSet, NodeDescriptor *to, ParameterSet *inAttrPSet, Simulation* sim)
{
      int myRank = sim->getRank();
      const int density = 1;                         // added by Jizhu Lu on 01/30/2006
      Edge* edge = et->getEdge();
      edge->initialize(edgeInit);
         
      Granule* fromGran = sim->getGranule(*from);
      Granule* toGran = sim->getGranule(*to);
      fromGran->addConnection(toGran);
      toGran->addComputeCost(density, et->getComputeCost());   

      // Necessary due to addConnection in repertoire.
      // Also needed due to internalExecute of C_edgeset, and others..
      edge->setPreNode(from);
      edge->setPostNode(to);
   
      GridLayerDescriptor *toGld, *fromGld;
      toGld = to->getGridLayerDescriptor();
      fromGld = from->getGridLayerDescriptor();
      if (sim->isEdgeRelationalDataEnabled()) {
	if (toGld != _to || fromGld!=_from) {
	  _to = toGld;
	  _from = fromGld;
	  _lcr = findLeastCommonRepertoire(from, to);
	}
	_lcr->addConnection(edge);
      }
}

void GranuleConnector::variableToNodeSet(
   VariableDescriptor* source, NodeSet* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   std::vector<NodeDescriptor*> nodes;
   destination->getNodes(nodes);
   std::vector<NodeDescriptor*>::iterator it = nodes.begin(), 
      end = nodes.end();
   std::unique_ptr<ParameterSet> outAttrPSet;
   source->getVariable()->getOutAttrParameterSet(outAttrPSet);
   outAttrPSet->set(*sourceOutAttr);
   std::unique_ptr<ParameterSet> inAttrPSet;
   (*it)->getGridLayerDescriptor()->getNodeType()->getInAttrParameterSet(std::move(inAttrPSet));
   inAttrPSet->set(*destinationInAttr);

   Granule *fromGran, *toGran;
   
   fromGran = sim->getGranule(*source);

   Node* node;
   for (; it != end; ++it) {
      toGran = sim->getGranule(**it);
      fromGran->addConnection(toGran);
   }
}

void GranuleConnector::variableToEdgeSet(
   VariableDescriptor* source, EdgeSet* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   std::vector<Edge*>& edges = destination->getEdges();
   std::vector<Edge*>::iterator it = edges.begin(), end = edges.end();
   std::unique_ptr<ParameterSet> outAttrPSet;
   source->getVariable()->getOutAttrParameterSet(outAttrPSet);
   outAttrPSet->set(*sourceOutAttr);
   std::unique_ptr<ParameterSet> inAttrPSet;
   (*it)->getInAttrParameterSet(inAttrPSet);
   inAttrPSet->set(*destinationInAttr);

   Granule *fromGran, *toGran;
   
   fromGran = sim->getGranule(*source);

   for (; it != end; ++it) {
      toGran = sim->getGranule(*((*it)->getPostNode()));
      fromGran->addConnection(toGran);
   }
}

void GranuleConnector::variableToVariable(
   VariableDescriptor* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   std::unique_ptr<ParameterSet> outAttrPSet;
   source->getVariable()->getOutAttrParameterSet(outAttrPSet);
   outAttrPSet->set(*sourceOutAttr);
   std::unique_ptr<ParameterSet> inAttrPSet;
   destination->getVariable()->getInAttrParameterSet(inAttrPSet);
   inAttrPSet->set(*destinationInAttr);
   
   Granule* fromGran = sim->getGranule(*source);
   Granule* toGran = sim->getGranule(*destination);
   fromGran->addConnection(toGran);      
}

void GranuleConnector::nodeSetToVariable(
   NodeSet* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   std::vector<NodeDescriptor*> nodes;
   source->getNodes(nodes);
   std::vector<NodeDescriptor*>::iterator it = nodes.begin(), end = nodes.end();
   std::unique_ptr<ParameterSet> outAttrPSet;
   (*it)->getGridLayerDescriptor()->getNodeType()->getOutAttrParameterSet(std::move(outAttrPSet));
   outAttrPSet->set(*sourceOutAttr);
   std::unique_ptr<ParameterSet> inAttrPSet;
   destination->getVariable()->getInAttrParameterSet(inAttrPSet);
   inAttrPSet->set(*destinationInAttr);

   Granule *fromGran, *toGran;

   toGran = sim->getGranule(*destination);
   Node* node;
   for (; it != end; ++it) {
     fromGran = sim->getGranule(**it);
     fromGran->addConnection(toGran);
   }
}

void GranuleConnector::edgeSetToVariable(
   EdgeSet* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   std::vector<Edge*>& edges = source->getEdges();
   std::vector<Edge*>::iterator it = edges.begin(), end = edges.end();
   std::unique_ptr<ParameterSet> outAttrPSet;
   (*it)->getOutAttrParameterSet(outAttrPSet);
   outAttrPSet->set(*sourceOutAttr);
   std::unique_ptr<ParameterSet> inAttrPSet;
   destination->getVariable()->getInAttrParameterSet(inAttrPSet);
   inAttrPSet->set(*destinationInAttr);

   Granule *fromGran, *toGran;
   
   toGran = sim->getGranule(*destination);

   for (; it != end; ++it) {
      fromGran = sim->getGranule(*((*it)->getPostNode()));
      fromGran->addConnection(toGran);
   }
}
