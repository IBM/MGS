// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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


void ComplexDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
