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

#ifndef VOLUMEGRANULEMAPPERTYPE_H
#define VOLUMEGRANULEMAPPERTYPE_H
#include "Copyright.h"

#include "GranuleMapperType.h"

#include <vector>

class Simulation;
class GranuleMapper;
class DataItem;
class NDPairList;

class VolumeGranuleMapperType : public GranuleMapperType
{
   public:
      VolumeGranuleMapperType(Simulation& s);
      void getGranuleMapper(std::vector<DataItem*> const & args, std::unique_ptr<GranuleMapper>&);
      virtual void duplicate(std::unique_ptr<GranuleMapperType>& dup) const;
      virtual ~VolumeGranuleMapperType();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
};
#endif
