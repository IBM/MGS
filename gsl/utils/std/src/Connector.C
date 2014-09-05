// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Connector.h"
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
//#include "Variable.h"
#include "VariableDescriptor.h"
#include "NDPairList.h"
#include "NodeSet.h"
#include "EdgeSet.h"

#include <list>


Connector::Connector()
   : _from(0),_to(0), _lcr(0)
{
}

Connector::~Connector()
{
}

Repertoire* Connector::findLeastCommonRepertoire(
   NodeDescriptor *from, NodeDescriptor* to)
{
   Grid *fromGrid = from->getGridLayerDescriptor()->getGrid();
   Grid *toGrid = to->getGridLayerDescriptor()->getGrid();
   return findLeastCommonRepertoire(fromGrid, toGrid);
}

Repertoire* Connector::findLeastCommonRepertoire(Grid* fromGrid, Grid* toGrid)
{
   Repertoire *current;
   Repertoire *fromRep = fromGrid->getParentRepertoire();
   Repertoire *toRep = toGrid->getParentRepertoire();

   std::list<Repertoire*> fromPath, toPath;

   // build Repertoire path from leaf to root for "from" Node
   for(current = fromRep; current!=0; current = current->getParentRepertoire())
      fromPath.push_back(current);

   // build Repertoire path from leaf to root for "to" Node
   for(current = toRep; current!=0; current = current->getParentRepertoire())
      toPath.push_back(current);

   // start at root and work toward leaves while they both match
   // keep last common Repertoire node
   Repertoire *common = 0;
   std::list<Repertoire*>::reverse_iterator fi;
   std::list<Repertoire*>::reverse_iterator fBegin = fromPath.rbegin();
   std::list<Repertoire*>::reverse_iterator fEnd = fromPath.rend();
   std::list<Repertoire*>::reverse_iterator ti;
   std::list<Repertoire*>::reverse_iterator tBegin = toPath.rbegin();
   std::list<Repertoire*>::reverse_iterator tEnd = toPath.rend();
   for (fi=fBegin, ti=tBegin; fi!=fEnd && ti!=tEnd; ++fi, ++ti) {
      if (*fi == *ti)
         common = *fi;
      else
         break;
   }
   return common;
}

void Connector::constantToVariable(
   Constant* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr)
{
  if (destination->getVariable()) {
    std::auto_ptr<ParameterSet> outAttrPSet;
    source->getOutAttrParameterSet(outAttrPSet);
    outAttrPSet->set(*sourceOutAttr);
    std::auto_ptr<ParameterSet> inAttrPSet;
    destination->getVariable()->getInAttrParameterSet(inAttrPSet);
    inAttrPSet->set(*destinationInAttr);
    source->addPostVariable(destination, outAttrPSet.get());
    destination->getVariable()->addPreConstant(source, inAttrPSet.get());
  }
}

void Connector::constantToNodeSet(
   Constant* source, NodeSet* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr)
{
   std::vector<NodeDescriptor*> nodes;
   destination->getNodes(nodes);
   std::vector<NodeDescriptor*>::iterator it = nodes.begin(), 
      end = nodes.end();
   std::auto_ptr<ParameterSet> outAttrPSet;
   source->getOutAttrParameterSet(outAttrPSet);
   outAttrPSet->set(*sourceOutAttr);
   std::auto_ptr<ParameterSet> inAttrPSet;
   Node* node;
   for (; it != end; ++it) {      
      // @TODO Distributed local filter
      if ((*it)->getNode()) {     // added by Jizhu Lu on 12/04/2005
	(*it)->getGridLayerDescriptor()->getNodeType()->getInAttrParameterSet(inAttrPSet);
	inAttrPSet->set(*destinationInAttr);
	node = (*it)->getNode();
	source->addPostNode((*it), outAttrPSet.get());
	node->addPreConstant(source, inAttrPSet.get());
      }
   }
}

void Connector::constantToNode(
   Constant* source, NodeDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr)
{
   Node* node;
   if (node=destination->getNode()) {
     std::auto_ptr<ParameterSet> outAttrPSet;
     source->getOutAttrParameterSet(outAttrPSet);
     outAttrPSet->set(*sourceOutAttr);
     std::auto_ptr<ParameterSet> inAttrPSet;  
     destination->getGridLayerDescriptor()->getNodeType()->getInAttrParameterSet(inAttrPSet);
     inAttrPSet->set(*destinationInAttr);
     source->addPostNode(node, outAttrPSet.get());
     node->addPreConstant(source, inAttrPSet.get());
   }
}

void Connector::constantToEdgeSet(
   Constant* source, EdgeSet* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr)
{
   std::vector<Edge*>& edges = destination->getEdges();
   std::vector<Edge*>::iterator it = edges.begin(), end = edges.end();
   std::auto_ptr<ParameterSet> outAttrPSet;
   source->getOutAttrParameterSet(outAttrPSet);
   outAttrPSet->set(*sourceOutAttr);
   std::auto_ptr<ParameterSet> inAttrPSet;
   for (; it != end; ++it) {
     (*it)->getInAttrParameterSet(inAttrPSet);
     inAttrPSet->set(*destinationInAttr);
     source->addPostEdge(*it, outAttrPSet.get());
     (*it)->addPreConstant(source, inAttrPSet.get());
   }
}
