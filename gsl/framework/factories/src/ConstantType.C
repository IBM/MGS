// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

void ConstantType::getInstance(std::unique_ptr<DataItem> & adi, 
			       std::vector<DataItem*> const * args, 
			       LensContext* c)
{
   ConstantDataItem* cdi = new ConstantDataItem;

   std::unique_ptr<Constant> ac;
   getConstant(ac);
   ac->initialize(c, *args);
   cdi->setConstant(ac);
   adi.reset(cdi);
}

void ConstantType::getInstance(std::unique_ptr<DataItem> & adi, 
			       const NDPairList& ndplist,
			       LensContext* c)
{
   ConstantDataItem* cdi = new ConstantDataItem;

   std::unique_ptr<Constant> ac;
   getConstant(ac);
   ac->initialize(ndplist);
   cdi->setConstant(ac);
   adi.reset(cdi);  
}


ConstantType::~ConstantType()
{
}
