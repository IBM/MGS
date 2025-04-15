// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "VariableDataItem.h"
#include "VariableInstanceAccessor.h"
#include "Variable.h"


const char* VariableDataItem::_type = "VARIABLE";


VariableDataItem::VariableDataItem()
   : _data(new VariableInstanceAccessor())
{
}

VariableDataItem::VariableDataItem(Variable* data)
{
   _data->setVariable(data);
}

VariableDataItem::~VariableDataItem()
{
}

void VariableDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new VariableDataItem(*this));
}

const char* VariableDataItem::getType() const
{
   return _type;
}

VariableInstanceAccessor* VariableDataItem::getVariable(Error* error) const
{
   return _data;
}

void VariableDataItem::setVariable(VariableInstanceAccessor* v , Error* error)
{
   _data = v;
}

std::string VariableDataItem::getString(Error* error) const
{
   return "";
}

std::vector<Triggerable*> VariableDataItem::getTriggerables()
{
   std::vector<Triggerable*> retVal;
   if (_data->getVariable()){
       retVal.push_back(_data->getVariable());
   }
   return retVal;
}
