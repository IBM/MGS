// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
#if defined(REUSE_NODEACCESSORS) and defined(TRACK_SUBARRAY_SIZE)
      Node* getSharedNode();
      void setSharedNode(Node*);
#endif
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
#if defined(REUSE_NODEACCESSORS) and defined(TRACK_SUBARRAY_SIZE)
      Node* _sharedNode;
#endif
};

#endif
