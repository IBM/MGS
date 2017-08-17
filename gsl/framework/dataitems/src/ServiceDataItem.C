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
void ServiceDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
