// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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

void ParameterSetType::duplicate(std::auto_ptr<DataType>& rv) const
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
