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

class GridGranuleMapper : public GranuleMapperBase
{

   public:
      GridGranuleMapper(Simulation& sim, std::vector<DataItem*> const & args);
      
      virtual Granule* getGranule(const NodeDescriptor& node);     
      virtual Granule* getGranule(const VariableDescriptor&) {
         assert(false);
         return 0;
      }
      virtual Granule* getGranule(unsigned gid) {
         return &(_granules[gid]);
      }
      virtual void addGranule(Granule*, unsigned) {
         assert(false);
      }
      virtual void getGranules(NodeSet& nodeSet,
			       GranuleSet& granuleSet);
      virtual std::string getName() {return _description;}
      virtual ~GridGranuleMapper();

   private:
      Simulation& _sim;
      std::string _description;
};
#endif
