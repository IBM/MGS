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
      virtual void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      virtual const char* getType() const;

   public:
      virtual InstanceFactory* getInstanceFactory() const;
      virtual void setInstanceFactory(InstanceFactory* s);
      std::string getString(Error* error=0) const;

};
#endif
