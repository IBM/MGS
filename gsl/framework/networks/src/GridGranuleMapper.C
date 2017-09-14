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

#include "GridGranuleMapper.h"
#include "VolumeOdometer.h"
#include "NodeDescriptor.h"
#include "ConnectionIncrement.h"
#include "NodeSet.h"
#include "Simulation.h"
#include "DistributableCompCategoryBase.h"
#include "DataItem.h"
#include "StringDataItem.h"
#include "ArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "NumericDataItem.h"
#include "FunctorDataItem.h"
#include "LayoutFunctor.h"
#include "Simulation.h"
#include "VectorOstream.h"
#include "Grid.h"

#include <iostream>
#include <cassert>
#include <algorithm>

GridGranuleMapper::GridGranuleMapper(Simulation& sim, std::vector<DataItem*> const & args)
   : GranuleMapperBase(), _sim(sim), _description("")
{
   if (args.size() != 3) {
      std::cerr<<"GridGranuleMapper accepts 3 arguments!"<<std::endl;
      exit(-1);
   }
   std::vector<DataItem*>::const_iterator iter = args.begin();

   StringDataItem* descriptionDI = dynamic_cast<StringDataItem*>(*iter);
   if (descriptionDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to StringDataItem failed on GridGranuleMapper!"<<std::endl;
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
     std::cerr << "numGranule = " << numGranules << ", numProcesses = " << _sim.getNumProcesses() << std::endl;
     std::cerr <<"Number of grid nodes does not equal number of processes on GridGranuleMapper!"<<std::endl;
     exit(-1);
   }
   _granules.resize(numGranules); // new Granules allocated here

   ++iter;
   IntArrayDataItem* densityDI = dynamic_cast<IntArrayDataItem*>(*iter);
   std::vector<int> density;
   if (densityDI != 0) {
     // Get density vector from int array
     density  =  *(densityDI->getIntVector());
   }     
   else {
     std::cerr<<"Dynamic cast of DataItem to IntArrayDataItem failed on VolumeGranuleMapper!"<<std::endl;
     exit(-1);
   }

   Grid g(dimensions);
   ConnectionIncrement* computeCost = new ConnectionIncrement;
   int uniformDensity = 0;
   if (density.size() == 1) {
      uniformDensity = density[0];
   } 
   for (unsigned i=0; i<numGranules; ++i) {
     std::vector<double>& coords = _granules[i].getModifiableGranuleCoordinates();
     std::vector<int> cds;
     g.getNodeCoords(i,cds);
     assert(cds.size()==dimensions.size());
     for (int j=0; j<cds.size(); ++j) {
       coords.push_back(double(cds[j])/double(dimensions[j]));
     }
     if (uniformDensity) {
       _granules[i].addComputeCost(uniformDensity, computeCost);
     } else {
       _granules[i].addComputeCost(density[i % density.size()], computeCost);
     }
   }
}

Granule* GridGranuleMapper::getGranule(const NodeDescriptor& node)
{
   return &(_granules[node.getNodeIndex()]);
}

void GridGranuleMapper::getGranules(
   NodeSet& nodeSet, GranuleSet& granuleSet)
{

  std::vector<NodeDescriptor*> nodes;
  nodeSet.getNodes(nodes);
  std::vector<NodeDescriptor*>::iterator iter, end=nodes.end();
  for (iter=nodes.begin(); iter!=end; ++iter) {
    granuleSet.insert(&(_granules[(*iter)->getNodeIndex()]));
  }
}

GridGranuleMapper::~GridGranuleMapper()
{
}
