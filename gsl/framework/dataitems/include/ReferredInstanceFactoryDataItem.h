// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef REFERREDINSTANCEFACTORYDATAITEM_H
#define REFERREDINSTANCEFACTORYDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

class InstanceFactory;

class ReferredInstanceFactoryDataItem : public InstanceFactoryDataItem
{
   protected:
      InstanceFactory *_instanceFactory;

   public:
      static const char* _type;

      ReferredInstanceFactoryDataItem& operator=(const ReferredInstanceFactoryDataItem& DI);
      ReferredInstanceFactoryDataItem& assign(const ReferredInstanceFactoryDataItem& DI);

      // Constructors
      ReferredInstanceFactoryDataItem(InstanceFactory *instanceFactory = 0);
      ReferredInstanceFactoryDataItem(const ReferredInstanceFactoryDataItem& DI);

      // Destructor
      virtual ~ReferredInstanceFactoryDataItem();

      // Utility methods
      virtual void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      virtual const char* getType() const;

      virtual InstanceFactory* getInstanceFactory() const;
      virtual void setInstanceFactory(InstanceFactory* s);

};
#endif
