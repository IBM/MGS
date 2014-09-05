// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef FFTGRANULEMAPPER2_H
#define FFTGRANULEMAPPER2_H
#include "Copyright.h"

#include "GranuleMapperBase.h"
#include "VolumeDivider.h"

#include <string>
#include <list>
#include <vector>
#include <deque>
#include <cassert>
#include <map>

class DataItem;
class Simulation;
class NodeDescriptor;
class Granule;
class VariableDescriptor;
class ConnectionIncrement;

class FFTGranuleMapper : public GranuleMapperBase
{
   public:
      FFTGranuleMapper(Simulation& sim, std::vector<DataItem*> const & args);
      
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
      //virtual unsigned getNumberOfGranules() {return _volumeDivider.getNumberOfPieces();}
      virtual unsigned getDefaultNumberOfGranules();
      virtual ~FFTGranuleMapper();

   private:
      VolumeDivider _volumeDivider;

      void setGranules(std::vector<int> const & density, ConnectionIncrement* computeCost);
      unsigned getSubPencil(const std::vector<int>& coordinates) const;
      void getPencilProjectionDims(std::vector<int>& coords, std::vector<int>& pencilProjDims);
      void getPencilProjectionOffsets(std::vector<int>& coords, std::vector<int>& pencilProjOffsets);
      virtual Granule* getGranule(std::vector<int>& coordinates);

      Simulation& _sim;
      std::string _description;
      int _nPencilDivs;
      std::map<unsigned, VolumeDivider*> _volumeSubdividersMap;
      std::map<std::string, VolumeDivider*> _dimsMap;
};
#endif
