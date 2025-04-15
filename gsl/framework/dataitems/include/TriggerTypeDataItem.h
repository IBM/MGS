// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;

      void setTriggerType(TriggerType*);
      TriggerType* getTriggerType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
