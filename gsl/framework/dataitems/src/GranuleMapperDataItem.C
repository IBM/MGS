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
void GranuleMapperDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
