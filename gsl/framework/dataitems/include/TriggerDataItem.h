// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TRIGGERDATAITEM_H
#define TRIGGERDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class Trigger;

class TriggerDataItem : public DataItem
{
   private:
      Trigger *_trigger;

   public:
      static const char* _type;

      virtual TriggerDataItem& operator=(const TriggerDataItem& DI);

      // Constructors
      TriggerDataItem(Trigger *trigger = 0);
      TriggerDataItem(const TriggerDataItem& DI);

      // Destructor
      ~TriggerDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Trigger* getTrigger() const;
      void setTrigger(Trigger* t);

};
#endif
