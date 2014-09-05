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

#include "ServiceType.h"
#include "LensType.h"
#include "DataType.h"
#include <string>
#include <memory>

void ServiceType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new ServiceType(*this));
}

std::string ServiceType::getDescriptor() const
{
   return "Service";
}

ServiceType::~ServiceType() 
{
}
