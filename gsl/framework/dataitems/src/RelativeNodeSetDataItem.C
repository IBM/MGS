// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "RelativeNodeSetDataItem.h"
#include "C_relative_nodeset.h"

// Type
const char* RelativeNodeSetDataItem::_type = "C_RELATIVE_NODE_SET";

// Constructors
RelativeNodeSetDataItem::RelativeNodeSetDataItem() 
   : _relativeNodeset(0)
{
}

RelativeNodeSetDataItem::RelativeNodeSetDataItem(
   std::unique_ptr<C_relative_nodeset> relativeNodeset)
{
   _relativeNodeset = relativeNodeset.release();
}


RelativeNodeSetDataItem::RelativeNodeSetDataItem(const RelativeNodeSetDataItem& DI)
{
   if (DI._relativeNodeset) {
      _relativeNodeset = DI._relativeNodeset->duplicate();
   }
}


// utility methods
void RelativeNodeSetDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new RelativeNodeSetDataItem(*this)));
}


RelativeNodeSetDataItem& RelativeNodeSetDataItem::operator=(const RelativeNodeSetDataItem& DI)
{
   _relativeNodeset = DI.getRelativeNodeSet();
   return(*this);
}


const char* RelativeNodeSetDataItem::getType() const
{
   return _type;
}


// Singlet methods

C_relative_nodeset* RelativeNodeSetDataItem::getRelativeNodeSet() const
{
   return _relativeNodeset;
}


void RelativeNodeSetDataItem::setRelativeNodeSet(C_relative_nodeset* rns)
{
   delete _relativeNodeset;
   if (rns) {
      _relativeNodeset = rns->duplicate();
   }
}


RelativeNodeSetDataItem::~RelativeNodeSetDataItem()
{
   delete _relativeNodeset;
}
