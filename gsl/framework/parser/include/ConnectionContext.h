// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CONNECTIONCONTEXT_H
#define CONNECTIONCONTEXT_H
#include "Copyright.h"

#include <list>
#include "SymbolTable.h"

#include "rndm.h"
#include <vector>

class NodeSet;
class NodeDescriptor;
class Edge;
class NodeType;
class EdgeType;
class ParameterSet;

/* Assumptions
The controlling Connector functor may do as much work as desired or call 
upon other functors to do it.
First, the connector functor sets the NodeSets.
Each time the Connector functor receives a valid pair of nodes it increments 
currentSamples.
currentSamples is set to zero before any functors are called.

Setting nodes is done by a SampFctr2 and/or SampFctr1's, if they are called. 
Otherwise, by the Connector functor.

SampFctr2:
Dependencies: sourceSet and destinationSet nodesets, currentSample, 
and its initialization parameters (which may indicate the stopping criterion, 
like number of samples)
Production:  sourceNode, destinationNode

SampFctr1:
Dependencies: all Node and NodeSet pointers, currentSample, and initialization 
parameters
Production: first non-null Node* based on corresponding NodeSet*

edgeType is set before ParameterSets are set.

Parameter sets are set by the Connector functor or by calling other functors

*/

class ConnectionContext
{
   public:
      enum Responsibility{_SOURCE, _DEST, _BOTH};

      ConnectionContext();
      ConnectionContext(const ConnectionContext&);
      void reset();

      ConnectionContext* parent;
      NodeSet* sourceSet;
      NodeSet* destinationSet;
#if defined(SUPPORT_MULTITHREAD_CONNECTION)
      /* only one is used
       *  use destinationNodes, if we iterate through source-nodes first
       *  use sourceNodes, if we iterate through dest-nodes first
       * */
      std::vector<NodeDescriptor*> destinationNodes;
      std::vector<NodeDescriptor*> sourceNodes;
#endif
      /* (sourceNode, destinationNode) is the node pair to be returned for connection */
      /*
       * restart = true if we're actually reset from the beginning
       * done    = true if we have reached to the end, i.e. traverse all the nodes 
       */
      NodeDescriptor* sourceNode;
      NodeDescriptor* destinationNode;
      NodeDescriptor* sourceRefNode;
      NodeDescriptor* destinationRefNode;
      EdgeType* edgeType;
      ParameterSet* edgeInitPSet;
      ParameterSet* inAttrPSet;
      ParameterSet* outAttrPSet;
      Responsibility current;
      int currentSample;
      bool restart;
      bool done;
};
#endif
