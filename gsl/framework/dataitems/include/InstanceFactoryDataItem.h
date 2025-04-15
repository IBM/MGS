// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef INSTANCEFACTORYDATAITEM_H
#define INSTANCEFACTORYDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class InstanceFactory;
class InstanceFactoryDataItem : public DataItem
{

   protected:
      InstanceFactory *_instanceFactory;

   public:
      static const char* _type;

      virtual InstanceFactoryDataItem& operator=(const InstanceFactoryDataItem& DI);

      // Constructors
      InstanceFactoryDataItem(InstanceFactory *instanceFactory = 0);
      InstanceFactoryDataItem(const InstanceFactoryDataItem& DI);

      // Destructor
      virtual ~InstanceFactoryDataItem();

      // Utility methods
      virtual void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      virtual const char* getType() const;

   public:
      virtual InstanceFactory* getInstanceFactory() const;
      virtual void setInstanceFactory(InstanceFactory* s);
      std::string getString(Error* error=0) const;

};
#endif
