// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GranuleMapper_H
#define GranuleMapper_H
#include "Copyright.h"

#include "GranuleSet.h"
#include <string>
#include <memory>

class NodeDescriptor;
class GridLayerDescriptor;
class Granule;
class Graph;
class NodeSet;
class Variable;
class VariableDescriptor;


class GranuleMapper
{
   public:
      virtual Granule* getGranule(const VariableDescriptor& variable) = 0;
      virtual Granule* getGranule(const NodeDescriptor& node) = 0;
      virtual Granule* getGranule(unsigned vid) = 0;
      virtual void addGranule(Granule*, unsigned) = 0;
      virtual void getGranules(NodeSet& nodeSet,
			       GranuleSet& granuleSet) = 0;
      virtual void setGraphId(unsigned& current) = 0;
      virtual void initializeGraph(Graph* graph) = 0;
      virtual void setGlobalGranuleIds(unsigned& id) = 0;
      virtual unsigned getIndex() = 0;
      virtual void setIndex(unsigned index) = 0;
      virtual void duplicate(std::unique_ptr<GranuleMapper>&& dup) const = 0;      
      virtual std::string getName() = 0;
      virtual unsigned getNumberOfGranules() = 0;
      virtual ~GranuleMapper(){}
};

#endif
