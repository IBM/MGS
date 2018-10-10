// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
