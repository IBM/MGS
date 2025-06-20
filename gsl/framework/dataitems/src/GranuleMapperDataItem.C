// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GranuleMapperDataItem.h"
#include "GranuleMapper.h"

// Type
const char* GranuleMapperDataItem::_type = "GRANULEMAPPER";

// Constructors
GranuleMapperDataItem::GranuleMapperDataItem(GranuleMapper *granuleMapper) 
   : _granuleMapper(granuleMapper)
{
}


GranuleMapperDataItem::GranuleMapperDataItem(const GranuleMapperDataItem& DI)
{
   _granuleMapper = DI._granuleMapper;
}


// Utility methods
void GranuleMapperDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new GranuleMapperDataItem(*this));
}


GranuleMapperDataItem& GranuleMapperDataItem::operator=(const GranuleMapperDataItem& DI)
{
   _granuleMapper = DI.getGranuleMapper();
   return(*this);
}


const char* GranuleMapperDataItem::getType() const
{
   return _type;
}


// Singlet methods

GranuleMapper* GranuleMapperDataItem::getGranuleMapper() const
{
   return _granuleMapper;
}


void GranuleMapperDataItem::setGranuleMapper(GranuleMapper* t)
{
   _granuleMapper = t;
}


GranuleMapperDataItem::~GranuleMapperDataItem()
{
}
