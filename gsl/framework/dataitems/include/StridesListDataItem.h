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

#ifndef STRIDESLISTDATAITEM_H
#define STRIDESLISTDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

#include "StridesList.h"
//class StridesList;

class StridesListDataItem : public DataItem
{
   private:
      StridesList *_stridesList;

   public:
      static const char* _type;

      virtual StridesListDataItem& operator=(const StridesListDataItem& DI);

      // Constructors
      StridesListDataItem();
      StridesListDataItem(const StridesListDataItem& DI);

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      StridesList * getStridesList() const;
      void setStridesList(StridesList* sl);
      ~StridesListDataItem();

};
#endif
