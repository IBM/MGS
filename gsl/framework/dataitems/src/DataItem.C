// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "DataItem.h"
#include <iostream>
#include <sstream>

std::ostream& operator<<(std::ostream& os, DataItem& di)
{
   os<<di.getType()<<"["<<di.getString()<<"]";
   return os;
}


std::string DataItem::getString(Error* error) const
{
   return "";
}


DataItem::DataItem()
{
}


DataItem::~DataItem()
{
}
