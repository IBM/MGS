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

#ifndef NodeInstanceAccessor_H
#define NodeInstanceAccessor_H
#include "Copyright.h"

#include "Node.h"
#include "ShallowArray.h"
#include "NodeDescriptor.h"

#include <deque>
#include <vector>
#include <set>
#include <cassert>

class Constant;
class Edge;
class Variable;
class Simulation;
class NodeDescriptor;
class Publisher;
class GridLayerData;
class GridLayerDescriptor;


class NodeInstanceAccessor : public NodeDescriptor
{

   public:
      NodeInstanceAccessor();
      virtual GridLayerData* getGridLayerData() const; 
      virtual GridLayerDescriptor* getGridLayerDescriptor() const;
      virtual void setGridLayerData(GridLayerData* gridLayerData);
      virtual int getDensityIndex() const;
      virtual void getNodeCoords(std::vector<int> & coords) const;
      virtual void getNodeCoords(ShallowArray<unsigned, 3, 2>& coords) const;
      virtual void getNodeCoords2Dim(int& x, int& y) const;
      virtual Node* getNode();
      virtual void setNode(Node*);
      virtual int getNodeIndex() const;
      virtual void setNodeIndex(int pos);
      virtual int getIndex() const;
      virtual void setIndex(int pos);
      virtual int getGlobalIndex() const;

      virtual ~NodeInstanceAccessor();

      virtual Publisher* getPublisher() {
	assert(_node);
	return _node->getPublisher();
      }
      virtual const char* getServiceName(void* data) const {
	return _node->getServiceName(data);
      }
      virtual const char* getServiceDescription(void* data) const {
	return _node->getServiceDescription(data);
      }

   protected:
      GridLayerData* _gridLayerData;
      Node* _node;
      int _nodeIndex;
      int _index;
};

#endif
