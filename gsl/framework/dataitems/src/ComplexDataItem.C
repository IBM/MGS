// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ComplexDataItem.h"
#include <iostream>
#include <sstream>

const char* ComplexDataItem::_type = "COMPLEX";

ComplexDataItem::ComplexDataItem()
:_complexType("")
{
}


ComplexDataItem::ComplexDataItem(std::string &complexType)
:_complexType(complexType)
{
}


ComplexDataItem& ComplexDataItem::operator=(const ComplexDataItem& DI)
{
   if (_complexType!="")
      std::cerr <<"Warning: Changing type of ComplexDataItem from "<<_complexType
         << " to " << DI._complexType<<"!"<<std::endl;
   _complexType = DI._complexType;
   _members = *(DI.getMembers());
   return(*this);
}


ComplexDataItem::ComplexDataItem(const ComplexDataItem& DI)
{
   _members = DI._members;
   _complexType = DI._complexType;
}


void ComplexDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new ComplexDataItem(*this)));
}


const char* ComplexDataItem::getType() const
{
   return _type;
}


std::string ComplexDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value<<_complexType<<"{";
   for (std::map<std::string, DataItem*>::const_iterator iter = _members.begin(); iter != _members.end(); iter++) {
      if (iter != _members.begin()) str_value<<",";
      str_value<<(*iter).first.c_str()<<":"<<(*iter).second;
   }
   str_value<<"}";
   return str_value.str();
}


const std::map<std::string, DataItem* >* ComplexDataItem::getMembers() const
{
   return &_members;
}


std::map<std::string, DataItem* >* ComplexDataItem::getModifiableMembers()
{
   return &_members;
}


const std::string ComplexDataItem::getComplexType()
{
   return _complexType;
}


void ComplexDataItem::setComplexType(std::string &complexType)
{
   _complexType = complexType;
}
