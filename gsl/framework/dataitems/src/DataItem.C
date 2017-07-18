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
