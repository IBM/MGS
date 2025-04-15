// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CONSTANTTYPEDATAITEM_H
#define CONSTANTTYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

#include <memory>

class ConstantType;

class ConstantTypeDataItem : public InstanceFactoryDataItem
{
   private:
      ConstantType *_data;

   public:
      static const char* _type;

      virtual ConstantTypeDataItem& operator=(const ConstantTypeDataItem& DI);

      // Constructors
      ConstantTypeDataItem(ConstantType *data = 0);
      ConstantTypeDataItem(const ConstantTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;

      void setConstantType(ConstantType*);
      ConstantType* getConstantType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
