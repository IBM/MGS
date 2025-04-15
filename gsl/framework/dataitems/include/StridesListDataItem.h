// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      StridesList * getStridesList() const;
      void setStridesList(StridesList* sl);
      ~StridesListDataItem();

};
#endif
