// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SERVICEDATAITEM_H
#define SERVICEDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class Service;

class ServiceDataItem : public DataItem
{
   private:
      Service *_service;

   public:
      static const char* _type;

      virtual ServiceDataItem& operator=(const ServiceDataItem& DI);

      // Constructors
      ServiceDataItem(Service *service = 0);
      ServiceDataItem(const ServiceDataItem& DI);

      // Destructor
      ~ServiceDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Service* getService() const;
      void setService(Service* s);

};
#endif
