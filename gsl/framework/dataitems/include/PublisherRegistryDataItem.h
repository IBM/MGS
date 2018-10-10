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

#ifndef PUBLISHERREGISTRYDATAITEM_H
#define PUBLISHERREGISTRYDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class PublisherRegistry;

class PublisherRegistryDataItem : public DataItem
{
   private:
      PublisherRegistry *_pubReg;

   public:
      static const char* _type;

      virtual PublisherRegistryDataItem& operator=(const PublisherRegistryDataItem& DI);

      // Constructors
      PublisherRegistryDataItem(PublisherRegistry *pubReg = 0);
      PublisherRegistryDataItem(const PublisherRegistryDataItem& DI);

      // Destructor
      ~PublisherRegistryDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      PublisherRegistry* getPublisherRegistry() const;
      void setPublisherRegistry(PublisherRegistry* pr);
      std::string getString(Error* error=0) const;

};
#endif
