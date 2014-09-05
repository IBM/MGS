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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Service* getService() const;
      void setService(Service* s);

};
#endif
