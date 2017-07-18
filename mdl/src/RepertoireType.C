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

#include "RepertoireType.h"
#include "LensType.h"
#include "DataType.h"
#include <string>
#include <memory>

void RepertoireType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new RepertoireType(*this));
}

std::string RepertoireType::getDescriptor() const
{
   return "Repertoire";
}

RepertoireType::~RepertoireType() 
{
}
