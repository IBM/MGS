// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "RankGranuleMapper.h"
#include "NodeDescriptor.h"
#include "ConnectionIncrement.h"
#include "NodeSet.h"
#include "Simulation.h"
#include "DistributableCompCategoryBase.h"
#include "DataItem.h"
#include "CustomStringDataItem.h"
#include "NumericDataItem.h"
#include "IntArrayDataItem.h"
#include "FunctorDataItem.h"
#include "LayoutFunctor.h"
#include "Simulation.h"
#include "VectorOstream.h"
#include "Grid.h"

#include <iostream>
#include <cassert>
#include <algorithm>

RankGranuleMapper::RankGranuleMapper(Simulation& sim, std::vector<DataItem*> const & args)
  : GranuleMapperBase(), _sim(sim), _description(""), _rank(0)
{
   if (args.size() != 3) {
     std::cerr<<"RankGranuleMapper accepts 3 arguments!"<<std::endl;
     exit(-1);
   }
   std::vector<DataItem*>::const_iterator iter = args.begin();

   CustomStringDataItem* descriptionDI = dynamic_cast<CustomStringDataItem*>(*iter);
   if (descriptionDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to CustomStringDataItem failed on RankGranuleMapper!"<<std::endl;
      exit(-1);
   }
   _description  = descriptionDI->getString();

   ++iter;
   IntArrayDataItem* dimensionsDI = dynamic_cast<IntArrayDataItem*>(*iter);
   if (dimensionsDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to IntArrayDataItem failed on GridGranuleMapper!"<<std::endl;
      exit(-1);
   }
   std::vector<int> const * v=dimensionsDI->getIntVector();
   std::vector<int> dimensions;
   dimensions.resize(v->size());
   std::copy(v->begin(), v->end(), dimensions.begin());

   std::vector<int>::iterator diter;
   int numGranules=1;
   for (diter=dimensions.begin(); diter!=dimensions.end(); ++diter) {
     numGranules*=*diter;
   }
   if (numGranules!=_sim.getNumProcesses()) {
     std::cerr<<"Number of grid nodes does not equal number of processes on RankGranuleMapper!"<<std::endl;
     exit(-1);
   }
   ++iter;
   NumericDataItem* rankDI = dynamic_cast<NumericDataItem*>(*iter);
   if (rankDI != 0) {
     _rank  =  rankDI->getInt();
   }     
   else {
     std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on VolumeGranuleMapper!"<<std::endl;
     exit(-1);
   }
   assert(_rank<numGranules);

   _granules.resize(numGranules); // new Granules allocated here
   Grid g(dimensions);
   ConnectionIncrement* computeCost = new ConnectionIncrement;
   for (unsigned i=0; i<numGranules; ++i) {
     std::vector<double>& coords = _granules[i].getModifiableGranuleCoordinates();
     std::vector<int> cds;
     g.getNodeCoords(i,cds);
     for (int j=0; j<cds.size(); ++j) {
       coords.push_back(double(cds[j])/double(dimensions[j]));
     }
     _granules[i].addComputeCost(1, computeCost);
   }
}

void RankGranuleMapper::getGranules(
   NodeSet& nodeSet, GranuleSet& granuleSet)
{
  std::vector<NodeDescriptor*> nodes;
  nodeSet.getNodes(nodes);
  int sz=nodes.size();
  for (int i=0; i<sz; ++i) {
    granuleSet.insert(&(_granules[_rank]));
  }
}

RankGranuleMapper::~RankGranuleMapper()
{
}
