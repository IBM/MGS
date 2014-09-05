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
