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
