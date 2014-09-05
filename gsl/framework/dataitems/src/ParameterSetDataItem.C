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

#include "ParameterSetDataItem.h"
#include "ParameterSet.h"

// Type
const char* ParameterSetDataItem::_type = "PARAMETER_SET";

// Constructors
ParameterSetDataItem::ParameterSetDataItem() 
   : _data(0)
{
}

ParameterSetDataItem::ParameterSetDataItem(std::auto_ptr<ParameterSet> data) 
{
   _data = data.release();
}


ParameterSetDataItem::ParameterSetDataItem(const ParameterSetDataItem& DI)
{
   std::auto_ptr<ParameterSet> pset;
   DI._data->duplicate(pset);
   _data = pset.release();
}


// Utility methods
void ParameterSetDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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


void ParameterSetDataItem::setParameterSet(std::auto_ptr<ParameterSet> & i)
{
   delete _data;
   _data = i.release();
}


ParameterSetDataItem::~ParameterSetDataItem()
{
   delete _data;
}
