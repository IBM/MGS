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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Trigger* getTrigger() const;
      void setTrigger(Trigger* t);

};
#endif
