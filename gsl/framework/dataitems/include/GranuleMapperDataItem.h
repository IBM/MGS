// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GRANULEMAPPERDATAITEM_H
#define GRANULEMAPPERDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class GranuleMapper;

class GranuleMapperDataItem : public DataItem
{
   private:
      GranuleMapper *_granuleMapper;

   public:
      static const char* _type;

      virtual GranuleMapperDataItem& operator=(const GranuleMapperDataItem& DI);

      // Constructors
      GranuleMapperDataItem(GranuleMapper *granuleMapper = 0);
      GranuleMapperDataItem(const GranuleMapperDataItem& DI);

      // Destructor
      ~GranuleMapperDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      GranuleMapper* getGranuleMapper() const;
      void setGranuleMapper(GranuleMapper* t);

};
#endif
