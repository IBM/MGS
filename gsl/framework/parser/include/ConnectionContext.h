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

#ifndef CONNECTIONCONTEXT_H
#define CONNECTIONCONTEXT_H
#include "Copyright.h"

#include <list>
#include "SymbolTable.h"

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
