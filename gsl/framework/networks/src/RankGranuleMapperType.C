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

#include "RankGranuleMapperType.h"
#include "RankGranuleMapper.h"
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

RankGranuleMapperType::RankGranuleMapperType(Simulation& s)
   : GranuleMapperType(s, "RankGranuleMapper", "Maps each grid layer by grid node index onto equal number of nprocs granules.")
{
}


void RankGranuleMapperType::getQueriable(
   std::auto_ptr<InstanceFactoryQueriable>& dup)
{
   dup.reset(new InstanceFactoryQueriable(this));
   Array<GranuleMapper*>::iterator it, end = _granuleMapperList.end();
   for (it = _granuleMapperList.begin(); it!=end; it++) {
      GranuleMapper* gm = (*it);
      GranuleMapperDataItem* gmdi = new GranuleMapperDataItem;
      gmdi->setGranuleMapper(gm);
      std::auto_ptr<DataItem> apdi(gmdi);
      DataItemQueriable* diq = new DataItemQueriable(apdi);
      diq->setName(gm->getName());
      std::auto_ptr<DataItemQueriable> apq(diq);
      dup->addQueriable(apq);
   }
   dup->setName(_name);
}


void RankGranuleMapperType::getGranuleMapper(std::vector<DataItem*> const & args, std::auto_ptr<GranuleMapper>& apgm)
{
   GranuleMapper* rval = new RankGranuleMapper(_sim, args);
   _granuleMapperList.push_back(rval);
   GranuleMapperDataItem* gmdi = new GranuleMapperDataItem;
   gmdi->setGranuleMapper(rval);
   _instances.push_back(gmdi);
   apgm.reset(rval);
}

void RankGranuleMapperType::duplicate(
   std::auto_ptr<GranuleMapperType>& dup) const
{
   dup.reset(new RankGranuleMapperType(*this));
}

RankGranuleMapperType:: ~RankGranuleMapperType()
{
}
