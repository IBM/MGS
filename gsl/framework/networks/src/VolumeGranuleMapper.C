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

#include "VolumeGranuleMapper.h"
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

#include <iostream>
#include <cassert>
#include <algorithm>

VolumeGranuleMapper::VolumeGranuleMapper(Simulation& sim, std::vector<DataItem*> const & args)
   : GranuleMapperBase(), _sim(sim), _description("")
{
   if (args.size() != 3 && args.size() != 4) {
      std::cerr<<"VolumeGranuleMapper accepts 3 or 4 arguments!"<<std::endl;
      exit(-1);
   }
   std::vector<DataItem*>::const_iterator iter = args.begin();

   StringDataItem* descriptionDI = dynamic_cast<StringDataItem*>(*iter);
   if (descriptionDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to StringDataItem failed on VolumeGranuleMapper! (first argument)"<<std::endl;
      exit(-1);
   }
   _description  = descriptionDI->getString();

   ++iter;
   IntArrayDataItem* dimensionsDI = dynamic_cast<IntArrayDataItem*>(*iter);
   if (dimensionsDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to IntArrayDataItem failed on VolumeGranuleMapper! (second argument)"<<std::endl;
      exit(-1);
   }
   std::vector<int> const * v=dimensionsDI->getIntVector();
   std::vector<int> dimensions;
   dimensions.resize(v->size());
   std::reverse_copy(v->begin(), v->end(), dimensions.begin());

   ++iter;
   IntArrayDataItem* densityDI = dynamic_cast<IntArrayDataItem*>(*iter);

   std::vector<int> density;

   if (densityDI != 0) {
     // Get density vector from int array
     density  =  *(densityDI->getIntVector());
   }     

   else {
     std::cerr<<"Dynamic cast of DataItem to IntArrayDataItem failed on VolumeGranuleMapper! (third argument)"<<std::endl;
     exit(-1);
   }

   unsigned numGranules;
   if (args.size() == 4) {
     ++iter;
     NumericDataItem* numGranulesDI = dynamic_cast<NumericDataItem*>(*iter);
     if (numGranulesDI == 0) {
       std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on VolumeGranuleMapper! (fourth argument)"<<std::endl;
       exit(-1);
     }
     numGranules=numGranulesDI->getInt();    
   }
   else numGranules=getDefaultNumberOfGranules();

   assert(numGranules>0);
   _volumeDivider.setUp(dimensions, numGranules);
   _granules.resize(numGranules); // new Granules allocated here
   for (unsigned i=0; i<numGranules; ++i) {
     std::vector<double>& coords = _granules[i].getModifiableGranuleCoordinates();
     _volumeDivider.getPieceCoordinates(i,coords);
   }
   ConnectionIncrement* computeCost = new ConnectionIncrement;
   setGranules(density, computeCost); // sets the cost associated with each granule
}

Granule* VolumeGranuleMapper::getGranule(const NodeDescriptor& node)
{
   Granule* granule;

   if (_volumeDivider.simpleMethod()) {
      granule = &(_granules[0]);
   } else {
      std::vector<int> coordinates;
      node.getNodeCoords(coordinates);
   
      // Returning a pointer in a vector is only ok if this vector doesn't get
      // changed later... The vector is constructed at the constructor and not
      // touched again.
      granule = &(_granules[_volumeDivider.getPiece(coordinates)]);
   }
   return granule;
}

void VolumeGranuleMapper::setGranules(std::vector<int> const & density, 
	     ConnectionIncrement* computeCost)
{
   std::vector<int> begin, end;

   unsigned size = 1;
   std::vector<int> const & dimensions = _volumeDivider.getDimensions();
   for (unsigned i = 0; i < dimensions.size(); ++i) {
      begin.push_back(0);
      end.push_back(dimensions[i] - 1);
      size *= dimensions[i];
   }

   VolumeOdometer vo(begin, end);

   int uniformDensity = 0;

   if (density.size() == 1) {
      uniformDensity = density[0];
   } 

   for (unsigned i = 0; i < size; ++i) {
      std::vector<int>& coords = vo.next();
      
      if (uniformDensity) {
	 _granules[_volumeDivider.getPiece(coords)].addComputeCost(uniformDensity, computeCost);
      } else {
	 _granules[_volumeDivider.getPiece(coords)].addComputeCost(density[i % density.size()], computeCost);
      }
   }
}

void VolumeGranuleMapper::getGranules(
   NodeSet& nodeSet, GranuleSet& granuleSet)
{
   // First find the beginning and end subGranule coords.
   std::vector<int> beginSubgranule;
   const std::vector<int>& beginCoords = nodeSet.getBeginCoords();
   
   std::vector<int>::const_iterator it, end = beginCoords.end();
   unsigned i;

   for (i = 0, it = beginCoords.begin(); it != end; ++it, ++i) {
      beginSubgranule.push_back(_volumeDivider.getSubPiece(i, *it));
   }
   
   std::vector<int> endSubgranule;
   const std::vector<int>& endCoords = nodeSet.getEndCoords();

   end = endCoords.end();
   for (i = 0, it = endCoords.begin(); it != end; ++it, ++i) {
      endSubgranule.push_back(_volumeDivider.getSubPiece(i, *it));
   }

   VolumeOdometer vo(beginSubgranule, endSubgranule);

   unsigned partition, size = beginSubgranule.size();   
   std::vector<int> const & strides = _volumeDivider.getStrides();
   std::vector<int>& coords = vo.look();
   for (; !vo.isRolledOver(); vo.next()) {
      partition = 0;
      for (unsigned i = 0; i < size; ++i) {
	 partition += coords[i] * strides[i];
      }   
      granuleSet.insert(&(_granules[partition]));
   }
}

unsigned VolumeGranuleMapper::getDefaultNumberOfGranules()
{
  return _sim.getNumProcesses();
}

VolumeGranuleMapper::~VolumeGranuleMapper()
{
}
