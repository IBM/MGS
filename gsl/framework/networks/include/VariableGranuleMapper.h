// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VariableGranuleMapper_H
#define VariableGranuleMapper_H
#include "Copyright.h"

#include "Granule.h"
#include "Variable.h"
#include "GranuleMapper.h"

#include <vector>
#include <deque>

class NodeDescriptor;
class GridLayerDescriptor;
class ConnectionIncrement;
class Simulation;
class VariableDescriptor;
//class Variable;

class VariableGranuleMapper : public GranuleMapper
{
   public:
      VariableGranuleMapper(Simulation* s);
      
      virtual Granule* getGranule(const NodeDescriptor& node);     
      virtual Granule* getGranule(const VariableDescriptor& variable) {
	return findGranule(variable.getVariableIndex());
      }
      virtual Granule* getGranule(unsigned gid) {
	return findGranule(gid);
      }     
      virtual unsigned getNumberOfGranules() {
	 return _granules.size();
      }     
      virtual void addGranule(Granule* granule, unsigned vid);
      virtual void getGranules(NodeSet& nodeSet, 
			       GranuleSet& granuleSet);
      virtual void setGraphId(unsigned& current);
      virtual void initializeGraph(Graph* graph);
      virtual void setGlobalGranuleIds(unsigned& id);
      virtual ~VariableGranuleMapper();
      virtual unsigned getIndex() {return _index;}
      virtual void setIndex(unsigned index) {_index=index;}
      virtual void duplicate(std::unique_ptr<GranuleMapper>&& dup) const {assert(0);}
      virtual std::string getName() {return _name;}
	
   private:
      Granule* findGranule(unsigned gid);
      std::map<unsigned, Granule*> _granules;
      Simulation * _sim;
      unsigned _index;
      static std::string _name;
};

#endif
