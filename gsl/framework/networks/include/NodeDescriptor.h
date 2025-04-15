// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NodeDescriptor_H
#define NodeDescriptor_H
#include "Copyright.h"

#include <vector>
#include "ShallowArray.h"
#include "Publishable.h"

class GridLayerData;
class GridLayerDescriptor;
class Node;

class NodeDescriptor : public Publishable
{
   public:
      virtual void setGridLayerData(GridLayerData* gridLayerData) = 0;
      virtual GridLayerData* getGridLayerData() const = 0;
      virtual GridLayerDescriptor* getGridLayerDescriptor() const = 0;
      virtual int getDensityIndex() const = 0;
      virtual void getNodeCoords(std::vector<int>& coords) const = 0;
      virtual void getNodeCoords(ShallowArray<unsigned, 3, 2>& coords) const=0;
      virtual void getNodeCoords2Dim(int& x, int& y) const = 0;
      virtual Node* getNode() = 0;
      virtual void setNode(Node*) = 0;
      virtual int getNodeIndex() const = 0;
      virtual void setNodeIndex(int pos) = 0;
      virtual int getIndex() const = 0;
      virtual void setIndex(int pos) = 0;
      virtual int getGlobalIndex() const = 0;
};


#endif
