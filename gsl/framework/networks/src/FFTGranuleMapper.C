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

#include "FFTGranuleMapper.h"
#include "VolumeOdometer.h"
#include "NodeDescriptor.h"
#include "ConnectionIncrement.h"
#include "NodeSet.h"
#include "Simulation.h"
#include "DistributableCompCategoryBase.h"
#include "DataItem.h"
#include "CustomStringDataItem.h"
#include "ArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "NumericDataItem.h"
#include "FunctorDataItem.h"
#include "LayoutFunctor.h"
#include "Simulation.h"
#include "VectorOstream.h"
#include "VolumeOdometer.h"

#include <iostream>
#include <cassert>
#include <algorithm>
#include <math.h>

#define DIM 3

FFTGranuleMapper::FFTGranuleMapper(Simulation& sim, std::vector<DataItem*> const & args)
   : GranuleMapperBase(), _sim(sim), _description(""), _nPencilDivs(0)
{
   if (args.size() != 3 && args.size() != 4) {
      std::cerr<<"FFTGranuleMapper accepts 3 or 4 arguments!"<<std::endl;
      exit(-1);
   }
   std::vector<DataItem*>::const_iterator iter = args.begin();

   CustomStringDataItem* descriptionDI = dynamic_cast<CustomStringDataItem*>(*iter);
   if (descriptionDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to CustomStringDataItem failed on FFTGranuleMapper!"<<std::endl;
      exit(-1);
   }
   _description  = descriptionDI->getString();

   ++iter;
   IntArrayDataItem* dimensionsDI = dynamic_cast<IntArrayDataItem*>(*iter);
   if (dimensionsDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to IntArrayDataItem failed on FFTGranuleMapper!"<<std::endl;
      exit(-1);
   }
   std::vector<int> const * v=dimensionsDI->getIntVector();
   std::vector<int> dimensions;
   dimensions.resize(v->size());
   std::reverse_copy(v->begin(), v->end(), dimensions.begin());
   assert(dimensions.size()==DIM);

   ++iter;
   IntArrayDataItem* densityDI = dynamic_cast<IntArrayDataItem*>(*iter);

   std::vector<int> density;

   if (densityDI != 0) {
     // Get density vector from int array
     density  =  *(densityDI->getIntVector());
   }     

   else {
     std::cerr<<"Dynamic cast of DataItem to IntArrayDataItem failed on FFTGranuleMapper!"<<std::endl;
     exit(-1);
   }

   unsigned numGranules;
   if (args.size() == 4) {
     ++iter;
     NumericDataItem* numGranulesDI = dynamic_cast<NumericDataItem*>(*iter);
     if (numGranulesDI == 0) {
       std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on FFTGranuleMapper!"<<std::endl;
       exit(-1);
     }
     numGranules=numGranulesDI->getInt();    
   }
   else numGranules=getDefaultNumberOfGranules();

   assert(numGranules>0);
   _volumeDivider.setUp(dimensions, numGranules);
   unsigned sz=DIM;
   assert(sz>1);

   _nPencilDivs = _volumeDivider.getDividers()[0]; // number of divisions in the pencil dimension

   _granules.resize(numGranules); // new Granules allocated here
   for (unsigned i=0; i<numGranules; ++i) {
     std::vector<double>& coords = _granules[i].getModifiableGranuleCoordinates();
     _volumeDivider.getPieceCoordinates(i,coords);
   }
   ConnectionIncrement* computeCost = new ConnectionIncrement;
   setGranules(density, computeCost); // sets the cost associated with each granule
}

Granule* FFTGranuleMapper::getGranule(const NodeDescriptor& node)
{
   Granule* granule;

   if (_volumeDivider.simpleMethod()) {
      granule = &(_granules[0]);
   } else {
      std::vector<int> coordinates;
      node.getNodeCoords(coordinates);
      granule=getGranule(coordinates);
   }
   return granule;
}

Granule* FFTGranuleMapper::getGranule(std::vector<int>& coordinates)
{
   Granule* granule;

   // Returning a pointer in a vector is only ok if this vector doesn't get
   // changed later... The vector is constructed at the constructor and not
   // touched again.
   // remap last dimension based on subdivider
   
   for (int i=0; i<DIM; ++i) assert(coordinates[i]<_volumeDivider.getDimensions()[i] && coordinates[i]>=0);
   unsigned subPencil = getSubPencil(coordinates);

   std::map<std::string, VolumeDivider*>::iterator dimsMapIter;
   std::map<unsigned, VolumeDivider*>::iterator volumeSubdividersIter;

   volumeSubdividersIter = _volumeSubdividersMap.find(subPencil);
   if (volumeSubdividersIter==_volumeSubdividersMap.end()) {
     std::vector<int> pencilProjDims;
     getPencilProjectionDims(coordinates, pencilProjDims); // note this method ignores first dimension of coords
     std::ostringstream os;
     os << pencilProjDims;
     dimsMapIter=_dimsMap.find(os.str());
     VolumeDivider* vdiv;
     if (dimsMapIter==_dimsMap.end()) {
       vdiv = new VolumeDivider();
       vdiv->setUp(pencilProjDims, _nPencilDivs);
       _dimsMap[os.str()]=vdiv;
     }
     else vdiv = dimsMapIter->second;
     _volumeSubdividersMap[subPencil]=vdiv;
     volumeSubdividersIter = _volumeSubdividersMap.find(subPencil);
   }
   std::vector<int> pencilProjOffsets;
   getPencilProjectionOffsets(coordinates, pencilProjOffsets); // note this method ignores first dimension of coordinates
   unsigned subPiece = volumeSubdividersIter->second->getPiece(pencilProjOffsets);
   int idx = subPencil+subPiece*_volumeDivider.getStrides()[0];
   granule = &(_granules[idx]);
   return granule;
}

void FFTGranuleMapper::setGranules(std::vector<int> const & density, 
	     ConnectionIncrement* computeCost)
{
   std::vector<int> begin, end;

   unsigned size = 1;
   
   std::vector<int> const & dimensions = _volumeDivider.getDimensions();
   for (unsigned i = 0; i < DIM; ++i) {
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

void FFTGranuleMapper::getGranules(
   NodeSet& nodeSet, GranuleSet& granuleSet)
{
  assert(0);
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

unsigned FFTGranuleMapper::getDefaultNumberOfGranules()
{
  return _sim.getNumProcesses();
}

unsigned FFTGranuleMapper::getSubPencil(const std::vector<int>& coordinates) const
{
  unsigned partition = 0;
  std::vector<int> const & strides = _volumeDivider.getStrides();

   if (_volumeDivider.simpleMethod()) {
      partition = 0;
   } else {
      unsigned sz = strides.size();
      for (unsigned i = 1; i < sz; ++i) {
         partition += _volumeDivider.getSubPiece(i, coordinates[i]) * strides[i];
      }   
   }
   return partition;  
}

void FFTGranuleMapper::getPencilProjectionDims(std::vector<int>& coords, std::vector<int>& pencilProjDims)
{
  pencilProjDims.clear();
  std::vector<int> const & dimensions = _volumeDivider.getDimensions();
  unsigned sz = DIM;
  if (_volumeDivider.simpleMethod()) {
    for (unsigned i=1; i<sz; ++i) {
      pencilProjDims.push_back(dimensions[i]);
    }
  }
  else { 
    std::vector<int> const & strides = _volumeDivider.getStrides();
    for (unsigned i = 1; i < sz; ++i) {
      pencilProjDims.push_back(_volumeDivider.getMinChunkSizes()[i] + 
			       (coords[i] < _volumeDivider.getCutOffs()[i] ? 0 : 1) );
    }
  }
}

void FFTGranuleMapper::getPencilProjectionOffsets(std::vector<int>& coords, std::vector<int>& pencilProjOffsets)
{
  pencilProjOffsets.clear();
  std::vector<int> const & dimensions = _volumeDivider.getDimensions();
  unsigned sz = DIM;
  if (_volumeDivider.simpleMethod()) {
    for (unsigned i = 1; i < sz; ++i) {
      pencilProjOffsets.push_back(coords[i]);
    }
  }
  else {
    std::vector<int> const & strides = _volumeDivider.getStrides();
    for (unsigned i = 1; i < sz; ++i) {
      int offset;
      if (coords[i] < _volumeDivider.getCutOffs()[i]) {
      offset = coords[i] % _volumeDivider.getMinChunkSizes()[i];
      } else {
	offset = (coords[i]-_volumeDivider.getCutOffs()[i]) % (_volumeDivider.getMinChunkSizes()[i]+1) ;
      }    
      pencilProjOffsets.push_back(offset);
    } 
  }
}
  
FFTGranuleMapper::~FFTGranuleMapper()
{
  std::map<std::string, VolumeDivider*>::iterator iter, end= _dimsMap.end();
  for (iter=_dimsMap.begin(); iter!=end; ++iter) {
    delete iter->second;
  }
}
