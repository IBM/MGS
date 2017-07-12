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
      virtual void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      virtual const char* getType() const;

      virtual InstanceFactory* getInstanceFactory() const;
      virtual void setInstanceFactory(InstanceFactory* s);

};
#endif
