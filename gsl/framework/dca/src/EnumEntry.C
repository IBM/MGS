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
