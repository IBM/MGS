// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GRANULEMAPPERTYPEDATAITEM_H
#define GRANULEMAPPERTYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

#include <memory>

class GranuleMapperType;

class GranuleMapperTypeDataItem : public InstanceFactoryDataItem
{
   private:
      GranuleMapperType *_data;

   public:
      static const char* _type;

      virtual GranuleMapperTypeDataItem& operator=(const GranuleMapperTypeDataItem& DI);

      // Constructors
      GranuleMapperTypeDataItem(GranuleMapperType *data = 0);
      GranuleMapperTypeDataItem(const GranuleMapperTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;

      void setGranuleMapperType(GranuleMapperType*);
      GranuleMapperType* getGranuleMapperType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
