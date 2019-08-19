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

#include "StridesList.h"
#include "StridesListDataItem.h"

// Type
const char* StridesListDataItem::_type = "STRIDES_LIST";

// Constructors
StridesListDataItem::StridesListDataItem() : _stridesList(0)
{
}


StridesListDataItem::StridesListDataItem(const StridesListDataItem& DI)
{
   _stridesList = new StridesList(DI._stridesList);
}


// Utility methods
void StridesListDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new StridesListDataItem(*this)));
}


StridesListDataItem& StridesListDataItem::operator=(const StridesListDataItem& DI)
{
   _stridesList = DI.getStridesList();
   return(*this);
}


const char* StridesListDataItem::getType() const
{
   return _type;
}


// Singlet methods

StridesList * StridesListDataItem::getStridesList() const
{
   return _stridesList;
}


void StridesListDataItem::setStridesList(StridesList * c)
{
   delete _stridesList;
   _stridesList = new StridesList(c);
}


StridesListDataItem::~StridesListDataItem()
{
   delete _stridesList;
}
