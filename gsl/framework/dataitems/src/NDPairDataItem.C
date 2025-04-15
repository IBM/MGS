// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NDPairDataItem.h"
#include "NDPair.h"

// Type
const char* NDPairDataItem::_type = "NDPAIR";

// Constructors

NDPairDataItem::NDPairDataItem()
   : _data(0)
{
}

NDPairDataItem::NDPairDataItem(std::unique_ptr<NDPair> data)
{
   _data = data.release();
}

NDPairDataItem::NDPairDataItem(const NDPairDataItem& DI)
{
   copyContents(DI);
}


// Utility methods
void NDPairDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new NDPairDataItem(*this));
}


NDPairDataItem& NDPairDataItem::operator=(const NDPairDataItem& DI)
{
   if (this == &DI) {
      return *this;
   }
   destructContents();
   copyContents(DI);
   return *this;
}

const char* NDPairDataItem::getType() const
{
   return _type;
}


NDPair* NDPairDataItem::getNDPair() const
{
   return _data;
}

void NDPairDataItem::releaseNDPair(std::unique_ptr<NDPair>& ndp)
{
   ndp.reset(_data);
   _data = 0;
}

void NDPairDataItem::setNDPair(std::unique_ptr<NDPair>& ndp)
{
   delete _data;
   _data = ndp.release();
}

NDPairDataItem::~NDPairDataItem()
{
   destructContents();
}

void NDPairDataItem::copyContents(const NDPairDataItem& DI)
{
   _data = new NDPair(*(DI._data));
}

void NDPairDataItem::destructContents()
{
   delete _data;
}

