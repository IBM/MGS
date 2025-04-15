// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ServiceDataItem.h"
#include "Service.h"

// Type
const char* ServiceDataItem::_type = "SERVICE";

// Constructors
ServiceDataItem::ServiceDataItem(Service *service) 
   : _service(service)
{
}


ServiceDataItem::ServiceDataItem(const ServiceDataItem& DI)
{
   _service = DI._service;
}


// Utility methods
void ServiceDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new ServiceDataItem(*this));
}


ServiceDataItem& ServiceDataItem::operator=(const ServiceDataItem& DI)
{
   _service = DI.getService();
   return(*this);
}


const char* ServiceDataItem::getType() const
{
   return _type;
}


// Singlet methods

Service* ServiceDataItem::getService() const
{
   return _service;
}


void ServiceDataItem::setService(Service* s)
{
   _service = s;
}


ServiceDataItem::~ServiceDataItem()
{
}
