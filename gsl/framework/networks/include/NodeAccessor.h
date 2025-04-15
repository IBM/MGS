// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NODEACCESSOR_H
#define NODEACCESSOR_H
#include "Copyright.h"

#include <memory>
#include <vector>
#include <string>

class Node;
class NodeDescriptor;
class GridLayerDescriptor;

class NodeAccessor
{
   public:
      virtual NodeDescriptor* getNodeDescriptor(std::vector<int> const & coords, int densityIndex) =0;
      virtual NodeDescriptor* getNodeDescriptor(int nodeIndex, int densityIndex) =0;
      virtual void duplicate(std::unique_ptr<NodeAccessor>&& r_aptr) const =0;
      virtual int getNbrUnits() =0;
      virtual GridLayerDescriptor* getGridLayerDescriptor() =0;
      virtual std::string getModelName() =0;
      virtual ~NodeAccessor() {}

      static const char* DENSITY_ERROR_MESSAGE;
      static const char* OFFSET_ERROR_MESSAGE;
};
#endif
