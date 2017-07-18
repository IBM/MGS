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
void IndexSetDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
