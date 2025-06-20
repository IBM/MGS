// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GRIDGRANULEMAPPER_H
#define GRIDGRANULEMAPPER_H
#include "Copyright.h"

#include "GranuleMapperBase.h"

#include <string>
#include <list>
#include <vector>
#include <deque>
#include <cassert>

class DataItem;
class Simulation;
class NodeDescriptor;
class Granule;
class VariableDescriptor;
class ConnectionIncrement;

class RankGranuleMapper : public GranuleMapperBase
{

   public:
      RankGranuleMapper(Simulation& sim, std::vector<DataItem*> const & args);
      
      virtual Granule* getGranule(const NodeDescriptor& node) {
         return &(_granules[_rank]);
      }
      virtual Granule* getGranule(const VariableDescriptor&) {
         assert(false);
         return 0;
      }
      virtual Granule* getGranule(unsigned gid) {
         return &(_granules[_rank]);
      }
      virtual void addGranule(Granule*, unsigned) {
         assert(false);
      }
      virtual void getGranules(NodeSet& nodeSet,
			       GranuleSet& granuleSet);
      virtual std::string getName() {return _description;}
      virtual ~RankGranuleMapper();

   private:
      Simulation& _sim;
      std::string _description;
      int _rank;
};
#endif
