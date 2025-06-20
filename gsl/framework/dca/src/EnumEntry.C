// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "EnumEntry.h"

EnumEntry::EnumEntry(std::string value, std::string description)
{
   _value = value;
   _description = description;
}


EnumEntry::EnumEntry(EnumEntry* ee)
: _value(ee->_value), _description(ee->_description)
{
}


std::string EnumEntry::getValue()
{
   return _value;
}


std::string EnumEntry::getDescription()
{
   return _description;
}


EnumEntry::~EnumEntry()
{
}
