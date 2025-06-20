// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "IndexSetDataItem.h"

// Type
const char* IndexSetDataItem::_type = "INDEX_SET";

// Constructors
IndexSetDataItem::IndexSetDataItem()
{
}


IndexSetDataItem::IndexSetDataItem(const IndexSetDataItem& DI)
{
   _set = DI._set;
}


// Utility methods
void IndexSetDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new IndexSetDataItem(*this));
}


IndexSetDataItem& IndexSetDataItem::operator=(const IndexSetDataItem& DI)
{
   _set = *(DI.getIndexSet());
   return(*this);
}


const char* IndexSetDataItem::getType() const
{
   return _type;
}


// Singlet methods

const IndexSet* IndexSetDataItem::getIndexSet() const
{
   return &_set;
}


void IndexSetDataItem::setIndexSet(IndexSet* i)
{
   _set = *i;
}
