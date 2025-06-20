// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "StructDataItem.h"
#include "Struct.h"

// Type
const char* StructDataItem::_type = "STRUCT";

// Constructors
StructDataItem::StructDataItem()
   : _data(0)
{
}

StructDataItem::StructDataItem(std::unique_ptr<Struct>& data)
{
   _data = data.release();
}

StructDataItem::~StructDataItem()
{
   destructOwnedHeap();
}

StructDataItem::StructDataItem(const StructDataItem& rv)
   : DataItem(rv), _data(0)
{
   copyOwnedHeap(rv);
}

// Utility methods
void StructDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new StructDataItem(*this));
}

StructDataItem& StructDataItem::operator=(const StructDataItem& rv)
{
   if (this != &rv) {
      DataItem::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

const char* StructDataItem::getType() const
{
   return _type;
}

// Singlet methods
Struct* StructDataItem::getStruct(Error* error) const
{
   return _data;
}

void StructDataItem::setStruct(std::unique_ptr<Struct>& s, Error* error)
{
   delete _data;
   _data = s.release();
}

std::string StructDataItem::getString(Error* error) const
{
   return "";
}

void StructDataItem::copyOwnedHeap(const StructDataItem& rv)
{
   if (rv._data) {
      std::unique_ptr<Struct> dup;
      rv._data->duplicate(std::move(dup));
      _data = dup.release();
   } else {
      _data = 0;
   }
}

void StructDataItem::destructOwnedHeap()
{
   delete _data;
}
