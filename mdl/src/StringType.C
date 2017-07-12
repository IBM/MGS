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

#include "StringType.h"
#include "DataType.h"
#include "Constants.h"
#include <string>
#include <sstream>
#include <memory>
#include <vector>

void StringType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new StringType(*this));
}

std::string StringType::getDescriptor() const
{
   return "String";
}

std::string StringType::getHeaderString(
   std::vector<std::string>& arrayTypeVec) const
{
   return "\"String.h\"";
}

std::string StringType::getDataItemFunctionString() const
{
   return "getLensString()";
}

StringType::~StringType() 
{
}
