// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Connector_H_
#define Connector_H_
#include "Copyright.h"

class EdgeType;
class NodeDescriptor;
class ParameterSet;
class Repertoire;
class Simulation;
class GridLayerDescriptor;
class Grid;
class Constant;
//class Variable;
class VariableDescriptor;                      // added 03/21/2006
class NDPairList;
class NodeSet;
class EdgeSet;


class Connector
{
   public:
      Connector();
      Repertoire* findLeastCommonRepertoire(NodeDescriptor* from, 
					    NodeDescriptor* to);
      Repertoire* findLeastCommonRepertoire(Grid* from, Grid* to);
      virtual void nodeToNode(
	 NodeDescriptor *from, ParameterSet *outAttrPSet, NodeDescriptor *to, 
	 ParameterSet *inAttrPSet, Simulation* sim) = 0;
      virtual void nodeToNodeWithEdge(
	 EdgeType *, ParameterSet *edgeInit, NodeDescriptor *from, 
	 ParameterSet *outAttrPSet, NodeDescriptor *to, 
	 ParameterSet *inAttrPSet, Simulation* sim) = 0;
      virtual ~Connector();
      void constantToVariable(Constant* source, VariableDescriptor* destination, 
			      NDPairList* sourceOutAttr, 
			      NDPairList* destinationInAttr);
      void constantToNodeSet(Constant* source, NodeSet* destination, 
			     NDPairList* sourceOutAttr, 
			     NDPairList* destinationInAttr, Simulation* sim);
      void constantToNode(Constant* source, NodeDescriptor* destination, NDPairList* sourceOutAttr, 
			  NDPairList* destinationInAttr, Simulation* sim);
      void constantToEdgeSet(Constant* source, EdgeSet* destination, 
			     NDPairList* sourceOutAttr, 
			     NDPairList* destinationInAttr);
      virtual void variableToNodeSet(VariableDescriptor* source, NodeSet* destination, 
				     NDPairList* sourceOutAttr, 
				     NDPairList* destinationInAttr, Simulation* sim) = 0;
      virtual void variableToEdgeSet(VariableDescriptor* source, EdgeSet* destination, 
				     NDPairList* sourceOutAttr, 
				     NDPairList* destinationInAttr, Simulation* sim) = 0;
      virtual void variableToVariable(VariableDescriptor* source, VariableDescriptor* destination, 
				      NDPairList* sourceOutAttr, 
				      NDPairList* destinationInAttr, Simulation* sim) = 0;
      virtual void nodeSetToVariable(NodeSet* source, VariableDescriptor* destination, 
				     NDPairList* sourceOutAttr, 
				     NDPairList* destinationInAttr, Simulation* sim) = 0;
      virtual void edgeSetToVariable(EdgeSet* source, VariableDescriptor* destination, 
				     NDPairList* sourceOutAttr, 
				     NDPairList* destinationInAttr, Simulation* sim) = 0;

   protected:
      GridLayerDescriptor *_from, *_to;
      Repertoire *_lcr;             // least common repertoire;
};
#endif
