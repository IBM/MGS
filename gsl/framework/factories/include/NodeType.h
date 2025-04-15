// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NODETYPE_H
#define NODETYPE_H
#include "Copyright.h"

#include <memory>
#include <string>


class ParameterSet;
class NodeAccessor;
class GridLayerDescriptor;
class ConnectionIncrement;

class NodeType
{
   public:
      virtual void getNodeAccessor(std::unique_ptr<NodeAccessor>&& r_aptr, GridLayerDescriptor* gridLayerDescriptor) =0;
      virtual void getInitializationParameterSet(std::unique_ptr<ParameterSet>&& r_aptr) =0;
      virtual void getInAttrParameterSet(std::unique_ptr<ParameterSet>&& r_aptr) =0;
      virtual void getOutAttrParameterSet(std::unique_ptr<ParameterSet>&& r_aptr) =0;
      virtual std::string getModelName() =0;
      virtual const char* c_str() const = 0;
      virtual ConnectionIncrement* getComputeCost() = 0;
      virtual ~NodeType() {}
};
#endif
