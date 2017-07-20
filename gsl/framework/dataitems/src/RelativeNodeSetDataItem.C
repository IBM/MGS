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
   std::auto_ptr<C_relative_nodeset> relativeNodeset)
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
void RelativeNodeSetDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
