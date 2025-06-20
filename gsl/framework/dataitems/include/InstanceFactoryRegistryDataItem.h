// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef INSTANCEFACTORYREGISTRYDATAITEM_H
#define INSTANCEFACTORYREGISTRYDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class InstanceFactoryRegistry;

class InstanceFactoryRegistryDataItem : public DataItem
{
   private:
      InstanceFactoryRegistry *_ifReg;

   public:
      static const char* _type;

      virtual InstanceFactoryRegistryDataItem& operator=(const InstanceFactoryRegistryDataItem& DI);

      // Constructors
      InstanceFactoryRegistryDataItem(InstanceFactoryRegistry *ifReg = 0);
      InstanceFactoryRegistryDataItem(const InstanceFactoryRegistryDataItem& DI);

      // Destructor
      ~InstanceFactoryRegistryDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      InstanceFactoryRegistry* getInstanceFactoryRegistry() const;
      void setInstanceFactoryRegistry(InstanceFactoryRegistry* pr);
      std::string getString(Error* error=0) const;

};
#endif
