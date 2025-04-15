// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ParameterSetDataItem.h"
#include "ParameterSet.h"

// Type
const char* ParameterSetDataItem::_type = "PARAMETER_SET";

// Constructors
ParameterSetDataItem::ParameterSetDataItem() 
   : _data(0)
{
}

ParameterSetDataItem::ParameterSetDataItem(std::unique_ptr<ParameterSet>& data) 
{
   _data = data.release();
}


ParameterSetDataItem::ParameterSetDataItem(const ParameterSetDataItem& DI)
{
   std::unique_ptr<ParameterSet> pset;
   DI._data->duplicate(std::move(pset));
   _data = pset.release();
}


// Utility methods
void ParameterSetDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new ParameterSetDataItem(*this));
}


ParameterSetDataItem& ParameterSetDataItem::operator=(const ParameterSetDataItem& DI)
{
   _data = DI.getParameterSet();
   return(*this);
}


const char* ParameterSetDataItem::getType() const
{
   return _type;
}


// Singlet methods

ParameterSet* ParameterSetDataItem::getParameterSet() const
{
   return _data;
}


void ParameterSetDataItem::setParameterSet(std::unique_ptr<ParameterSet> & i)
{
   delete _data;
   _data = i.release();
}


ParameterSetDataItem::~ParameterSetDataItem()
{
   delete _data;
}
