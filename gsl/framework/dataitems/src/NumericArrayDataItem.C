// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
