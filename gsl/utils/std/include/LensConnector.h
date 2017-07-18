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

#ifndef LensConnector_H_
#define LensConnector_H_
#include "Copyright.h"

#include "Connector.h"

class EdgeType;
class NodeDescriptor;
class ParameterSet;
class Repertoire;
class Simulation;
class GridLayerDescriptor;
class Grid;
class Constant;
//class Variable;
class VariableDescriptor;              // added on 03/21/2006
class NDPairList;
class NodeSet;
class EdgeSet;


class LensConnector : public Connector
{
   public:
      LensConnector();
      virtual void nodeToNode(NodeDescriptor *from, ParameterSet *outAttrPSet,
			      NodeDescriptor *to, ParameterSet *inAttrPSet, Simulation* sim);
      virtual void nodeToNodeWithEdge(
	 EdgeType *, ParameterSet *edgeInit, NodeDescriptor *from, 
	 ParameterSet *outAttrPSet, NodeDescriptor *to, 
	 ParameterSet *inAttrPSet, Simulation* sim);
      virtual ~LensConnector();
      virtual void variableToNodeSet(VariableDescriptor* source, NodeSet* destination, 
				     NDPairList* sourceOutAttr, 
				     NDPairList* destinationInAttr, Simulation* sim);
      virtual void variableToEdgeSet(VariableDescriptor* source, EdgeSet* destination, 
				     NDPairList* sourceOutAttr, 
				     NDPairList* destinationInAttr, Simulation* sim);
      virtual void variableToVariable(VariableDescriptor* source, VariableDescriptor* destination, 
				      NDPairList* sourceOutAttr, 
				      NDPairList* destinationInAttr, Simulation* sim);
      virtual void nodeSetToVariable(NodeSet* source, VariableDescriptor* destination, 
				     NDPairList* sourceOutAttr, 
				     NDPairList* destinationInAttr, Simulation* sim);
      virtual void edgeSetToVariable(EdgeSet* source, VariableDescriptor* destination, 
				     NDPairList* sourceOutAttr, 
				     NDPairList* destinationInAttr, Simulation* sim);
      

};
#endif
