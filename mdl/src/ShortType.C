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

#include "ShortType.h"
#include "SignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

void ShortType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new ShortType(*this));
}

std::string ShortType::getDescriptor() const
{
   return "short";
}

std::string ShortType::getCapitalDescriptor() const
{
   return "Short";
}

ShortType::~ShortType() 
{
}
