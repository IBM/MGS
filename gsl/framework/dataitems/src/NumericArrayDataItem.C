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

#include "NumericArrayDataItem.h"

NumericArrayDataItem::NumericArrayDataItem()
{
}


NumericArrayDataItem::~NumericArrayDataItem()
{
}


NumericArrayDataItem::NumericArrayDataItem(std::vector<int> const &dimensions)
: ArrayDataItem(dimensions)
{
}


NumericArrayDataItem& NumericArrayDataItem::operator=(const NumericArrayDataItem& DI)
{
   assign(DI);
   return(*this);
}
