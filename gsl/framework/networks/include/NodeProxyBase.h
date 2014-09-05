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

#ifndef NodeProxyBase_H
#define NodeProxyBase_H
#include "Copyright.h"

#include "Node.h"
#include "NodeRelationalDataUnit.h"

#include <cassert>
#include <iostream>

class Constant;
class Edge;
class Variable;
class VariableDescriptor;
class GridLayerData;
class NodeDescriptor;
class Publisher;
class Simulation;

class NodeProxyBase : public Node
{
   public:
      NodeProxyBase();
      virtual ~NodeProxyBase(); 

      // Functions that are normally implemented by the 
      // generated modules. They are not required for proxy nodes. 
      // [begin]
      virtual void initialize(ParameterSet* initPSet) {
	 assert(false);
      }

      virtual void getInitializationParameterSet(
	 std::auto_ptr<ParameterSet>& initPSet) const {
	 assert(false);
      }

      virtual void getInAttrParameterSet(
	 std::auto_ptr<ParameterSet>& CG_castedPSet) const {
	 assert(false);
      }

      virtual void getOutAttrParameterSet(
	 std::auto_ptr<ParameterSet>& CG_castedPSet) const {
	 assert(false);
      }

      virtual void acceptService(Service* service, const std::string& name) {
	 assert(false);
      }

      virtual Publisher* getPublisher() {
	 assert(false);
	 return 0;
      }

      virtual const char* getServiceName(void* data) const {
	 assert(false);
	 return "";
      }

      virtual const char* getServiceDescription(void* data) const {
	 assert(false);
	 return "";
      }

      virtual const std::deque<Constant*>& getPreConstantList();
      virtual const std::deque<Edge*>& getPreEdgeList();
      virtual const std::deque<NodeDescriptor*>& getPreNodeList();
      virtual const std::deque<VariableDescriptor*>& getPreVariableList();
      virtual const std::deque<Edge*>& getPostEdgeList();
      virtual const std::deque<NodeDescriptor*>& getPostNodeList();
      virtual const std::deque<VariableDescriptor*>& getPostVariableList();

      // Node Descriptor implementation functions.
      virtual GridLayerData* getGridLayerData() const;
      virtual GridLayerDescriptor* getGridLayerDescriptor() const;
      virtual void setGridLayerData(GridLayerData* gridLayerData);
      virtual int getDensityIndex() const;
      virtual void getNodeCoords(std::vector<int> & coords) const;
      virtual void getNodeCoords(ShallowArray<unsigned, 3, 2>& coords) const;
      virtual void getNodeCoords2Dim(int& x, int& y) const;
      virtual Node* getNode() {
	 return this;
      }
      virtual void setNode(Node*);
      virtual int getNodeIndex() const;
      virtual void setNodeIndex(int pos);
      virtual int getIndex() const;
      virtual void setIndex(int pos);
 
      virtual void setNodeDescriptor(NodeDescriptor* nodeDescriptor) {
	_nodeInstanceAccessor = nodeDescriptor;
      }
      virtual NodeDescriptor* getNodeDescriptor() const {return _nodeInstanceAccessor;} 

      virtual int getPartition() const {
	 return _partition;
      }
      virtual void setPartition(int pos) {
	 _partition = pos;
      }
      virtual float getComputeCost() const { 
	 return 1;
      }
      virtual bool hasService() {
         return false;
      }
      virtual void addPreConstant(Constant* CG_constant, 
				  ParameterSet* CG_pset) {
	 assert(false);
      }

      virtual void addPreVariable(VariableDescriptor* CG_variable, 
				  ParameterSet* CG_pset) {
	 assert(false);
      }

      virtual void addPreEdge(Edge* CG_edge, ParameterSet* CG_pset) {
	 assert(false);
      }

      virtual void addPreNode(NodeDescriptor* CG_node, ParameterSet* CG_pset) {
	 assert(false);
      }


   protected:
      NodeRelationalDataUnit* _relationalInformation;

      void checkAndAddPreConstant(Constant* c);
      void checkAndAddPreEdge(Edge* e);
      void checkAndAddPreNode(NodeDescriptor* n);
      void checkAndAddPreVariable(VariableDescriptor* v);
      void checkAndAddPostEdge(Edge* e);
      void checkAndAddPostNode(NodeDescriptor* n);
      void checkAndAddPostVariable(VariableDescriptor* v);
      inline bool relationalDataEnabled();

      // Functions that are normally implemented by the 
      // generated modules. They are not required for regular nodes. 
      // [begin]
      virtual TriggerableBase::EventType createTriggerableCaller(
	 const std::string& name, NDPairList* ndpList, 
	 std::auto_ptr<TriggerableCaller>& triggerableCaller);
      // [end]

      NodeDescriptor* _nodeInstanceAccessor;
      int _partition;
};

#endif
