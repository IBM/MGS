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

#ifndef TRIGGERTYPEDATAITEM_H
#define TRIGGERTYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

#include <memory>

class TriggerType;

class TriggerTypeDataItem : public InstanceFactoryDataItem
{
   private:
      TriggerType *_data;

   public:
      static const char* _type;

      virtual TriggerTypeDataItem& operator=(const TriggerTypeDataItem& DI);

      // Constructors
      TriggerTypeDataItem(TriggerType *data = 0);
      TriggerTypeDataItem(const TriggerTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;

      void setTriggerType(TriggerType*);
      TriggerType* getTriggerType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
