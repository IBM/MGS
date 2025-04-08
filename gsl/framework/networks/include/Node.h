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

#ifndef NODE_H
#define NODE_H
#include "Copyright.h"

#include "ServiceAcceptor.h"
#include "TriggerableBase.h"
#include "NodeDescriptor.h"

#ifdef HAVE_MPI
#include "ConnectionIncrement.h"
#endif

#include <deque>
#include <vector>

class Constant;
class Edge;
class VariableDescriptor;
class ParameterSet;
class GridLayerDescriptor;
class GridLayerData;
class Simulation;
class NodeDescriptor;

class Node : public NodeDescriptor, public ServiceAcceptor, public TriggerableBase
{
   public:
      virtual void initialize(ParameterSet* initPSet) =0;

      virtual const std::deque<Edge*>& getPreEdgeList() =0;
      virtual const std::deque<Edge*>& getPostEdgeList() =0;
      virtual const std::deque<NodeDescriptor*>& getPreNodeList() = 0;
      virtual const std::deque<NodeDescriptor*>& getPostNodeList() = 0;
      virtual void addPostEdge(Edge* e, ParameterSet* OutAttrPSet)=0;
      virtual void addPostNode(NodeDescriptor* n, ParameterSet* OutAttrPSet) = 0;
      virtual void addPostVariable(VariableDescriptor* v, ParameterSet* OutAttrPSet) = 0;
      virtual void addPreConstant(Constant* c, ParameterSet* InAttrPSet) = 0;
      virtual void addPreEdge(Edge* e, ParameterSet* InAttrPSet)=0;
      virtual void addPreNode(NodeDescriptor* n, ParameterSet* InAttrPSet) = 0;
#if defined(REUSE_NODEACCESSORS) and defined(TRACK_SUBARRAY_SIZE)
      virtual void addPreNode_Dummy(NodeDescriptor* CG_node, ParameterSet* CG_pset, Simulation* sim, NodeDescriptor* nd_for_this_node) = 0; 
#endif
      virtual void addPreVariable(VariableDescriptor* v, ParameterSet* InAttrPSet) = 0;
      virtual ~Node() {}

      virtual void getInitializationParameterSet(
	 std::unique_ptr<ParameterSet>&& r_aptr) const = 0;
      virtual void getInAttrParameterSet(
	 std::unique_ptr<ParameterSet>&& r_aptr) const = 0;
      virtual void getOutAttrParameterSet(
	 std::unique_ptr<ParameterSet>&& r_aptr) const = 0;

//      virtual float getComputeCost() const = 0;
      virtual bool hasService() = 0;
};

#endif
