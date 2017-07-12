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

#include "PhaseDataItem.h"
#include "Phase.h"

// Type
const char* PhaseDataItem::_type = "PHASE";

// Constructors
PhaseDataItem::PhaseDataItem()
   : _data(0)
{
}

PhaseDataItem::PhaseDataItem(std::auto_ptr<Phase> data)
{
   _data = data.release();
}

PhaseDataItem::~PhaseDataItem()
{
   destructOwnedHeap();
}

PhaseDataItem::PhaseDataItem(const PhaseDataItem& rv)
   : DataItem(rv), _data(0)
{
   copyOwnedHeap(rv);
}

// Utility methods
void PhaseDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new PhaseDataItem(*this));
}

PhaseDataItem& PhaseDataItem::operator=(const PhaseDataItem& rv)
{
   if (this != &rv) {
      DataItem::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

const char* PhaseDataItem::getType() const
{
   return _type;
}

// Singlet methods
Phase* PhaseDataItem::getPhase(Error* error) const
{
   return _data;
}

void PhaseDataItem::setPhase(std::auto_ptr<Phase>& data, Error* error)
{
   delete _data;
   _data = data.release();
}

std::string PhaseDataItem::getString(Error* error) const
{
   return "";
}

void PhaseDataItem::copyOwnedHeap(const PhaseDataItem& rv)
{
   if (rv._data) {
      std::auto_ptr<Phase> dup;
      rv._data->duplicate(dup);
      _data = dup.release();
   } else {
      _data = 0;
   }
}

void PhaseDataItem::destructOwnedHeap()
{
   delete _data;
}
