// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
