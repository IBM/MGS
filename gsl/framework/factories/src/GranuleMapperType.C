// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GranuleMapperType.h"
#include "GranuleMapperDataItem.h"
#include "DataItem.h"
#include "GranuleMapper.h"
#include "SyntaxErrorException.h"
#include "NDPairList.h"
#include "Simulation.h"

GranuleMapperType::GranuleMapperType(Simulation& s, const std::string& name, const std::string& description)
   : InstanceFactory(), _name(name), _description(description), _sim(s)
{
}

void GranuleMapperType::getInstance(std::unique_ptr<DataItem> & adi, 
			      std::vector<DataItem*> const * args, 
			      LensContext* c)
{
   GranuleMapperDataItem* di = new GranuleMapperDataItem();
   std::unique_ptr<GranuleMapper> apgm;
   if (_sim.isGranuleMapperPass() || _sim.isCostAggregationPass()) {
     getGranuleMapper(*args, apgm);
     GranuleMapper* gm = apgm.get();
     di->setGranuleMapper(gm);
     gm->setIndex(_sim.getGranuleMapperCount());
     _sim.addGranuleMapper(apgm);
   }
   else {
     di->setGranuleMapper(_sim.getGranuleMapper(_sim.getGranuleMapperCount()));     
   }
   _sim.incrementGranuleMapperCount();
   adi.reset(di);
}

void GranuleMapperType::getInstance(std::unique_ptr<DataItem> & adi, 
			      const NDPairList& ndplist,
			      LensContext* c)
{
   throw SyntaxErrorException(
      "GranuleMappers can not be instantiated with Name-DataItem pair lists.");
}

GranuleMapperType::~GranuleMapperType() {
}

