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

#include "ConstantType.h"
#include "ConstantDataItem.h"
#include "Constant.h"
#include "DataItemQueriable.h"
#include "InstanceFactoryQueriable.h"
#include "NDPairList.h"

ConstantType::ConstantType()
   : InstanceFactory()
{
}

void ConstantType::getInstance(std::auto_ptr<DataItem> & adi, 
			       std::vector<DataItem*> const * args, 
			       LensContext* c)
{
   ConstantDataItem* cdi = new ConstantDataItem;

   std::auto_ptr<Constant> ac;
   getConstant(ac);
   ac->initialize(c, *args);
   cdi->setConstant(ac);
   adi.reset(cdi);
}

void ConstantType::getInstance(std::auto_ptr<DataItem> & adi, 
			       const NDPairList& ndplist,
			       LensContext* c)
{
   ConstantDataItem* cdi = new ConstantDataItem;

   std::auto_ptr<Constant> ac;
   getConstant(ac);
   ac->initialize(ndplist);
   cdi->setConstant(ac);
   adi.reset(cdi);  
}


ConstantType::~ConstantType()
{
}
