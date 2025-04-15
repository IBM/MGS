// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "CustomStringDataItem.h"
#include "CustomString.h"
#include <string>
#include <sstream>
#define DOUBLE_MANTISSA_MAX = 2251799813685250;
#define DOUBLE_MANTISSA_MIN (- DOUBLE_MANTISSA_MAX -1)

// Type
const char* CustomStringDataItem::_type = "STRING";

// Constructors
CustomStringDataItem::CustomStringDataItem(const std::string& data)
   : _data(data)
{
}

CustomStringDataItem::CustomStringDataItem(CustomString& data)
{
   _data = data.c_str();
}


CustomStringDataItem::CustomStringDataItem(const CustomStringDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void CustomStringDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new CustomStringDataItem(*this)));
}


CustomStringDataItem& CustomStringDataItem::operator=(const CustomStringDataItem& DI)
{
   _data = DI.getString();
   return(*this);
}


const char* CustomStringDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string CustomStringDataItem::getString(Error* error) const
{
   return _data;
}

CustomString CustomStringDataItem::getLensString(Error* error) const
{
   return CustomString(_data.c_str());
}


void CustomStringDataItem::setString(std::string i, Error* error)
{
   _data = i;
}
