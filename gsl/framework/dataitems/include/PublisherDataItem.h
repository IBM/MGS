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
