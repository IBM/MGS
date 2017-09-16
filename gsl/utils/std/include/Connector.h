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
