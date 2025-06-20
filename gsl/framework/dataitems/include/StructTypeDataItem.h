// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef STRUCTTYPEDATAITEM_H
#define STRUCTTYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

#include <memory>

class StructType;

class StructTypeDataItem : public InstanceFactoryDataItem
{
   private:
      StructType *_data;

   public:
      static const char* _type;

      virtual StructTypeDataItem& operator=(const StructTypeDataItem& DI);

      // Constructors
      StructTypeDataItem(StructType *data = 0);
      StructTypeDataItem(const StructTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;

      void setStructType(StructType*);
      StructType* getStructType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
