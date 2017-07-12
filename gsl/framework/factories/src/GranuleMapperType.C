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

void GranuleMapperType::getInstance(std::auto_ptr<DataItem> & adi, 
			      std::vector<DataItem*> const * args, 
			      LensContext* c)
{
   GranuleMapperDataItem* di = new GranuleMapperDataItem();
   std::auto_ptr<GranuleMapper> apgm;
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

void GranuleMapperType::getInstance(std::auto_ptr<DataItem> & adi, 
			      const NDPairList& ndplist,
			      LensContext* c)
{
   throw SyntaxErrorException(
      "GranuleMappers can not be instantiated with Name-DataItem pair lists.");
}

GranuleMapperType::~GranuleMapperType() {
}

