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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      InstanceFactoryRegistry* getInstanceFactoryRegistry() const;
      void setInstanceFactoryRegistry(InstanceFactoryRegistry* pr);
      std::string getString(Error* error=0) const;

};
#endif
