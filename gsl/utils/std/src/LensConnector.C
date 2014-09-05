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

#include "LensConnector.h"
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
#include "GridLayerData.h"
#include "CompCategory.h"
#include "NodeCompCategoryBase.h"
#include "VariableCompCategoryBase.h"

#include <list>

LensConnector::LensConnector()
{
}

LensConnector::~LensConnector()
{
}

void LensConnector::nodeToNode(NodeDescriptor *from, ParameterSet *outAttrPSet,
			       NodeDescriptor *to, ParameterSet *inAttrPSet, Simulation* sim)
{ 
#ifdef HAVE_MPI
   int myRank = sim->getRank();
   int fromPartitionId = sim->getGranule(*from)->getPartitionId();
   int toPartitionId = sim->getGranule(*to)->getPartitionId();

   if ( (fromPartitionId != myRank) && (toPartitionId != myRank) ){
      /* both nodes are not in current memory space: do nothing */
   }
   else if ( (fromPartitionId != myRank) && (toPartitionId == myRank) ) {
      /* "from" is not in current memory space but "to" is: 
	 check for "from" proxy, allocate if necessary, connect */
      if (from->getNode()==0) {
         from->getGridLayerData()->getNodeCompCategoryBase()->allocateProxy(fromPartitionId, from);
	 assert(from->getNode());
      }
      //from->getNode()->addPostNode(to, outAttrPSet); // not added yet because use not clear but cost is
      to->getNode()->addPreNode(from, inAttrPSet);
   }
   else if (fromPartitionId == myRank && toPartitionId != myRank) {
      /* "from" is in current memory space but "to" is not: 
	 add the "from" node to the node list which will 
	 send message to the memory space "to" is in */
      from->getGridLayerData()->getNodeCompCategoryBase()->addToSendMap(toPartitionId, from->getNode());
      //from->getNode()->addPostNode(to, outAttrPSet);
   }
   else{

#endif
      /* Both "from" and "to" are in current memory space: connect as usual */
      /* For the first pass, Computation cost including Computation Time, Memory, 
         Communication Overhead of the two nodes need to be added together as
         the total cost of this connection.  -- Jizhu Lu  01/24/06 */

      from->getNode()->addPostNode(to, outAttrPSet);
      to->getNode();
      to->getNode()->addPreNode(from, inAttrPSet);

#ifdef  HAVE_MPI
   }
#endif

}

void LensConnector::nodeToNodeWithEdge(
   EdgeType *et, ParameterSet *edgeInit, NodeDescriptor *from, 
   ParameterSet *outAttrPSet, NodeDescriptor *to, ParameterSet *inAttrPSet, Simulation* sim)
{
#ifdef HAVE_MPI
   int myRank = sim->getRank();
   int fromPartitionId = sim->getGranule(*from)->getPartitionId();
   int toPartitionId = sim->getGranule(*to)->getPartitionId();

   if ( (fromPartitionId != myRank) && (toPartitionId != myRank) ){
      /* both nodes are not in current memory space: do nothing */
   }
   else if ( (fromPartitionId != myRank) && (toPartitionId == myRank) ) {
      /* "from" is not in current memory space but "to" is: 
	 check for "from" proxy, allocate if necessary, connect */
      if (from->getNode()==0) {
         from->getGridLayerData()->getNodeCompCategoryBase()->allocateProxy(fromPartitionId, from);
      }

      Edge* edge = et->getEdge();
      edge->initialize(edgeInit);
   
      edge->setPreNode(from);
      from->getNode()->addPostEdge(edge, outAttrPSet);
      edge->setPostNode(to);
      to->getNode()->addPreEdge(edge, inAttrPSet);
   
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
   else if (fromPartitionId == myRank && toPartitionId == myRank) {
#endif
      /* both "from" and "to" are in current memory space */
      Edge* edge = et->getEdge();
      edge->initialize(edgeInit);
   
      edge->setPreNode(from);
      from->getNode()->addPostEdge(edge, outAttrPSet);
      edge->setPostNode(to);
      to->getNode()->addPreEdge(edge, inAttrPSet);
   
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
#ifdef HAVE_MPI
   } else {
      // add the destination memory space ID to send map
      from->getGridLayerData()->getNodeCompCategoryBase()->addToSendMap(toPartitionId, from->getNode());
   }
#endif
      
}

void LensConnector::variableToNodeSet(
   VariableDescriptor* source, NodeSet* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
#ifdef HAVE_MPI
   int myRank = sim->getRank();
   int fromPartitionId = sim->getGranule(*source)->getPartitionId();
#endif
   std::vector<NodeDescriptor*> nodes;
   destination->getNodes(nodes);
   std::vector<NodeDescriptor*>::iterator it = nodes.begin(), end = nodes.end();
#ifdef HAVE_MPI
   if (fromPartitionId != myRank) {
     Variable* v=0;
     std::auto_ptr<ParameterSet> outAttrPSet;
     std::auto_ptr<ParameterSet> inAttrPSet;
     for (; it != end; ++it) {
       int toPartitionId = sim->getGranule(*(*it))->getPartitionId();
       if (toPartitionId == myRank) {
	 source->getVariableType()->allocateProxy(fromPartitionId, source);
	 v=source->getVariable();
	 v->getOutAttrParameterSet(outAttrPSet);
	 outAttrPSet->set(*sourceOutAttr);
	 (*it)->getGridLayerDescriptor()->getNodeType()->getInAttrParameterSet(inAttrPSet);
	 inAttrPSet->set(*destinationInAttr);
	 v->addPostNode(*it, outAttrPSet.get());
	 (*it)->getNode()->addPreVariable(v, inAttrPSet.get());
       }
     }
   }
   else if (nodes.size()>0) {
#endif
     std::auto_ptr<ParameterSet> outAttrPSet;
     source->getVariable()->getOutAttrParameterSet(outAttrPSet);
     outAttrPSet->set(*sourceOutAttr);
     std::auto_ptr<ParameterSet> inAttrPSet;

     (*it)->getGridLayerDescriptor()->getNodeType()->getInAttrParameterSet(inAttrPSet);
     inAttrPSet->set(*destinationInAttr);
    
     for (; it != end; ++it) {
#ifdef HAVE_MPI
       // @TODO Distributed local filter
       int toPartitionId = sim->getGranule(*(*it))->getPartitionId();
       if (toPartitionId == myRank) {
#endif
	 source->getVariable()->addPostNode(*it, outAttrPSet.get());
	 (*it)->getNode()->addPreVariable(source->getVariable(), inAttrPSet.get());

#ifdef HAVE_MPI
       } else {
	 // add the destination memory space ID to send map
	 source->getVariableType()->addToSendMap(toPartitionId, source->getVariable());
       }
     }
#endif

   } // ifdef HAVE_MPI: end of else; ifndef HAVE_MPI: end of for(; it != end; ++it)
}

void LensConnector::variableToEdgeSet(
   VariableDescriptor* source, EdgeSet* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
#ifdef HAVE_MPI
   int myRank = sim->getRank();
   int fromPartitionId = sim->getGranule(*source)->getPartitionId();
#endif
   std::vector<Edge*>& edges = destination->getEdges();
   std::vector<Edge*>::iterator it = edges.begin(), end = edges.end();
#ifdef HAVE_MPI
   if (fromPartitionId != myRank) {
     bool postEdge=false;
     for (; it != end; ++it) {
       if (*it) {
	 postEdge=true;
	 break;
       }
     }
     it = edges.begin();

     if (postEdge) {
       // create a proxy for the source variable in current memory space
       source->getVariableType()->allocateProxy(fromPartitionId, source);
     }
   }
   else {
#endif
     std::auto_ptr<ParameterSet> outAttrPSet;
     source->getVariable()->getOutAttrParameterSet(outAttrPSet);
     outAttrPSet->set(*sourceOutAttr);
     std::auto_ptr<ParameterSet> inAttrPSet;
     (*it)->getInAttrParameterSet(inAttrPSet);
     inAttrPSet->set(*destinationInAttr);
     for (; it != end; ++it) {
       if (*it) {
         source->getVariable()->addPostEdge(*it, outAttrPSet.get());
         (*it)->addPreVariable(source->getVariable(), inAttrPSet.get());
       }
#ifdef HAVE_MPI
       else {
	 // add the destination memory space ID to send map
	 assert(0);
	 //source->getVariableType()->addToSendMap(toPartitionId, source->getVariable());
       }
     }
#endif
   } // ifdef HAVE_MPI: end of else; ifndef HAVE_MPI: end of for(; it != end; ++it)
}


void LensConnector::variableToVariable(
   VariableDescriptor* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   Variable *from = source->getVariable();
   Variable *to = destination->getVariable();
#ifdef HAVE_MPI
   int myRank = sim->getRank();
   int fromPartitionId = sim->getGranule(*source)->getPartitionId();
   int toPartitionId = sim->getGranule(*destination)->getPartitionId();
   if (!from && !to) {
      // if both source and destination are not in current memory space do nothing
   }
   else if (from && to) {
#endif
      // if both source and destination are in current memory space do nothing
      std::auto_ptr<ParameterSet> outAttrPSet;
      from->getOutAttrParameterSet(outAttrPSet);
      outAttrPSet->set(*sourceOutAttr);
      std::auto_ptr<ParameterSet> inAttrPSet;
      destination->getVariable()->getInAttrParameterSet(inAttrPSet);
      inAttrPSet->set(*destinationInAttr);
      
      from->addPostVariable(destination->getVariable(), outAttrPSet.get());
      destination->getVariable()->addPreVariable(source->getVariable(), inAttrPSet.get());   
#ifdef HAVE_MPI
   } 
   else if (!from && to) {
      // create a proxy for the source variable in current memory space
      std::auto_ptr<ParameterSet> outAttrPSet;
      from->getOutAttrParameterSet(outAttrPSet);
      outAttrPSet->set(*sourceOutAttr);
      std::auto_ptr<ParameterSet> inAttrPSet;
      destination->getVariable()->getInAttrParameterSet(inAttrPSet);
      inAttrPSet->set(*destinationInAttr);
      
      from->addPostVariable(destination->getVariable(), outAttrPSet.get());
      destination->getVariable()->addPreVariable(source->getVariable(), inAttrPSet.get());   
   }
   else {
       // add the destination memory space ID to send map
      source->getVariableType()->addToSendMap(toPartitionId, from);
   }
#endif
}

void LensConnector::nodeSetToVariable(
   NodeSet* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   Variable* to = destination->getVariable();
   std::vector<NodeDescriptor*> nodes;
   source->getNodes(nodes);
   if (nodes.size()>0) {
     std::vector<NodeDescriptor*>::iterator it = nodes.begin(), end = nodes.end();
#ifdef HAVE_MPI
     int myRank = sim->getRank();
     int toPartitionId = sim->getGranule(*destination)->getPartitionId();
     if (to) {
#endif
       std::auto_ptr<ParameterSet> outAttrPSet;
       (*it)->getGridLayerDescriptor()->getNodeType()->getOutAttrParameterSet(outAttrPSet);
       outAttrPSet->set(*sourceOutAttr);
       std::auto_ptr<ParameterSet> inAttrPSet;
       to->getInAttrParameterSet(inAttrPSet);
       inAttrPSet->set(*destinationInAttr);
   
       Node* from;
       for (; it != end; ++it) {
         // @TODO Distributed local filter
         from = (*it)->getNode();
#ifdef HAVE_MPI
         int fromPartitionId = sim->getGranule(*(*it))->getPartitionId();
         if (fromPartitionId == myRank) {
#endif
	   assert(from);
	   from->addPostVariable(to, outAttrPSet.get());
	   to->addPreNode(*it, inAttrPSet.get());
#ifdef HAVE_MPI
         } else {
	   if (!from) {
	     // create a proxy for from in current memory space
	     //(*it)->getGridLayerData()->getNodeCompCategoryBase()->allocateProxy(fromPartitionId, from);
	     (*it)->getGridLayerData()->getNodeCompCategoryBase()->allocateProxy(fromPartitionId, *it);
	   }
	   from = (*it)->getNode();
	   from->getNode()->addPostVariable(to, outAttrPSet.get());
	   to->addPreNode(*it, inAttrPSet.get());
         }
#endif
       }
#ifdef HAVE_MPI
     } else {
       Node* from;
       for (; it != end; ++it) {
         from = (*it)->getNode();
         int fromPartitionId = sim->getGranule(*(*it))->getPartitionId();
         if (fromPartitionId == myRank) {
	   // add the destination memory space ID to send map
	   assert(from);
	   from->getGridLayerData()->getNodeCompCategoryBase()->addToSendMap(toPartitionId, from);
         }
       }
     }
#endif
   }
}

void LensConnector::edgeSetToVariable(
   EdgeSet* source, VariableDescriptor* destination, NDPairList* sourceOutAttr, 
   NDPairList* destinationInAttr, Simulation* sim)
{
   Variable* to = destination->getVariable();
   if (to) {
      std::vector<Edge*>& edges = source->getEdges();
      std::vector<Edge*>::iterator it = edges.begin(), end = edges.end();
      if (edges.size()>0) {
	std::auto_ptr<ParameterSet> outAttrPSet;
	(*it)->getOutAttrParameterSet(outAttrPSet);
	outAttrPSet->set(*sourceOutAttr);
	std::auto_ptr<ParameterSet> inAttrPSet;
	to->getInAttrParameterSet(inAttrPSet);
	inAttrPSet->set(*destinationInAttr);
	
	for (; it != end; ++it) {
	  if (*it) {
            (*it)->addPostVariable(to, outAttrPSet.get());
            to->addPreEdge(*it, inAttrPSet.get());
	  }
	}
      }
   }
}
