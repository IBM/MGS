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

#include "ParameterSetType.h"
#include "LensType.h"
#include "DataType.h"
#include <string>
#include <memory>

void ParameterSetType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new ParameterSetType(*this));
}

std::string ParameterSetType::getDescriptor() const
{
   return "ParameterSet";
}

bool ParameterSetType::shouldBeOwned() const
{
   return true;
}

ParameterSetType::~ParameterSetType() 
{
}
