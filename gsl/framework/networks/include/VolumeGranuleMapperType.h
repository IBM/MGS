// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
