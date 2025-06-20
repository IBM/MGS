// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "VolumeGranuleMapperType.h"
#include "VolumeGranuleMapper.h"
#include "Simulation.h"
#include "GranuleMapperDataItem.h"
#include "DataItem.h"
#include "Queriable.h"
#include "DataItemQueriable.h"
#include "InstanceFactoryQueriable.h"
#include "NDPairList.h"
#include "NDPairItemFinder.h"
#include "SyntaxErrorException.h"

#include <algorithm>
//#include <iostream>
//#include <sstream>

VolumeGranuleMapperType::VolumeGranuleMapperType(Simulation& s)
   : GranuleMapperType(s, "VolumeGranuleMapper", "Divides each grid layer by volume into nprocs granules.")
{
}


void VolumeGranuleMapperType::getQueriable(
   std::unique_ptr<InstanceFactoryQueriable>& dup)
{
   dup.reset(new InstanceFactoryQueriable(this));
   Array<GranuleMapper*>::iterator it, end = _granuleMapperList.end();
   for (it = _granuleMapperList.begin(); it!=end; it++) {
      GranuleMapper* gm = (*it);
      GranuleMapperDataItem* gmdi = new GranuleMapperDataItem;
      gmdi->setGranuleMapper(gm);
      std::unique_ptr<DataItem> apdi(gmdi);
      DataItemQueriable* diq = new DataItemQueriable(apdi);
      diq->setName(gm->getName());
      std::unique_ptr<DataItemQueriable> apq(diq);
      dup->addQueriable(apq);
   }
   dup->setName(_name);
}


void VolumeGranuleMapperType::getGranuleMapper(std::vector<DataItem*> const & args, std::unique_ptr<GranuleMapper>& apgm)
{
   GranuleMapper* rval = new VolumeGranuleMapper(_sim, args);
   _granuleMapperList.push_back(rval);
   GranuleMapperDataItem* gmdi = new GranuleMapperDataItem;
   gmdi->setGranuleMapper(rval);
   _instances.push_back(gmdi);
   apgm.reset(rval);
}

void VolumeGranuleMapperType::duplicate(
   std::unique_ptr<GranuleMapperType>& dup) const
{
   dup.reset(new VolumeGranuleMapperType(*this));
}

VolumeGranuleMapperType:: ~VolumeGranuleMapperType()
{
}
