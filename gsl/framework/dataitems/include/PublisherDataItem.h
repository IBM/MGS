// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PUBLISHERDATAITEM_H
#define PUBLISHERDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class Publisher;

class PublisherDataItem : public DataItem
{
   private:
      Publisher *_publisher;

   public:
      static const char* _type;

      virtual PublisherDataItem& operator=(const PublisherDataItem& DI);

      // Constructors
      PublisherDataItem(Publisher *publisher = 0);
      PublisherDataItem(const PublisherDataItem& DI);

      // Destructor
      ~PublisherDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Publisher* getPublisher() const;
      void setPublisher(Publisher* p);

};
#endif
