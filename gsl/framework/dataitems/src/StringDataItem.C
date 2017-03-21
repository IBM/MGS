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

#include "StringDataItem.h"
#include "String.h"
#include <string>
#include <sstream>
#define DOUBLE_MANTISSA_MAX = 2251799813685250;
#define DOUBLE_MANTISSA_MIN (- DOUBLE_MANTISSA_MAX -1)

// Type
const char* StringDataItem::_type = "STRING";

// Constructors
StringDataItem::StringDataItem(const std::string& data)
   : _data(data)
{
}

StringDataItem::StringDataItem(String& data)
{
   _data = data.c_str();
}


StringDataItem::StringDataItem(const StringDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void StringDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new StringDataItem(*this)));
}


StringDataItem& StringDataItem::operator=(const StringDataItem& DI)
{
   _data = DI.getString();
   return(*this);
}


const char* StringDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string StringDataItem::getString(Error* error) const
{
   return _data;
}

String StringDataItem::getLensString(Error* error) const
{
   return String(_data.c_str());
}


void StringDataItem::setString(std::string i, Error* error)
{
   _data = i;
}
