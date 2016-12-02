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
