// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "PhaseDataItem.h"
#include "Phase.h"

// Type
const char* PhaseDataItem::_type = "PHASE";

// Constructors
PhaseDataItem::PhaseDataItem()
   : _data(0)
{
}

PhaseDataItem::PhaseDataItem(std::unique_ptr<Phase>& data)
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
void PhaseDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
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

void PhaseDataItem::setPhase(std::unique_ptr<Phase>& data, Error* error)
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
      std::unique_ptr<Phase> dup;
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
