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

#include "LongType.h"
#include "SignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

void LongType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new LongType(*this));
}

std::string LongType::getDescriptor() const
{
   return "long";
}

std::string LongType::getCapitalDescriptor() const
{
   return "Long";
}

LongType::~LongType() 
{
}
