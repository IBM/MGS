// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VARIABLETYPEDATAITEM_H
#define VARIABLETYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

#include <memory>

class VariableType;

class VariableTypeDataItem : public InstanceFactoryDataItem
{
   private:
      VariableType *_data;

   public:
      static const char* _type;

      virtual VariableTypeDataItem& operator=(const VariableTypeDataItem& DI);

      // Constructors
      VariableTypeDataItem(VariableType *data = 0);
      VariableTypeDataItem(const VariableTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;

      void setVariableType(VariableType*);
      VariableType* getVariableType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
