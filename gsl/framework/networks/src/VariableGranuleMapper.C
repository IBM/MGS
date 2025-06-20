// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "VariableGranuleMapper.h"
#include "ConnectionIncrement.h"
#include "Simulation.h"
#include "NodeDescriptor.h"
//#include "CompCategory.h"
#include "DistributableCompCategoryBase.h"     // modified by Jizhu Lu on 02/16/2006
//#include "Graph.h"
#include <cassert>
#include <map>

std::string VariableGranuleMapper::_name="VariableGranuleMapper";


VariableGranuleMapper::VariableGranuleMapper(Simulation* s)
   : GranuleMapper(), _sim(s), _index(0)
{
   int density = 1;                   // added by Jizhu Lu on 01/30/2006
   std::map<unsigned, Granule*>::iterator it, end = _granules.end();
   ConnectionIncrement* computeCost;
   computeCost = new ConnectionIncrement;
   for (it = _granules.begin(); it != end; ++it) {
      (*it).second->addComputeCost(density, computeCost);   // added by Jizhu Lu on 01/30/2006
   }
}

Granule* VariableGranuleMapper::getGranule(const NodeDescriptor& node)
{
   assert(false);
   return 0;
}

void VariableGranuleMapper::getGranules(
   NodeSet& nodeSet, GranuleSet& granuleSet)
{
   assert(false);
}

void VariableGranuleMapper::addGranule(Granule* granule, unsigned vid) {
   _granules[vid] = granule;
}

void VariableGranuleMapper::setGraphId(unsigned& current)
{
   std::map<unsigned, Granule*>::iterator it, end = _granules.end();
   for (it = _granules.begin(); it != end; ++it) {
      (*it).second->setGraphId(current);
   }
}

void VariableGranuleMapper::initializeGraph(Graph* graph)
{
   std::map<unsigned, Granule*>::iterator it, end = _granules.end();
   for (it = _granules.begin(); it != end; ++it) {
      (*it).second->initializeGraph(graph);
   }
}

void VariableGranuleMapper::setGlobalGranuleIds(unsigned& id)
{
   std::map<unsigned, Granule*>::iterator it, end = _granules.end();
   for (it = _granules.begin(); it != end; ++it) {
      (*it).second->setPartitionId(0);                // added by Jizhu Lu on 12/12/2005 to fix a bug.
      (*it).second->setGlobalGranuleId(id++);
   }
}

Granule* VariableGranuleMapper::findGranule(unsigned id)
{
  std::map<unsigned, Granule*>::iterator iter=_granules.find(id);
  if (iter==_granules.end()) {
    std::cerr<<"Error! VariableGranuleMapper: variable or granule id not found in map!"<<std::endl;
    exit(0);
  }
  return iter->second;
}

VariableGranuleMapper::~VariableGranuleMapper()
{
}
