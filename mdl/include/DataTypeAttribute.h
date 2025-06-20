// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include <memory>
#ifndef DataTypeAttribute_H
#define DataTypeAttribute_H
#include "Mdl.h"

#include "AccessType.h"
#include "DataType.h"
#include "Attribute.h"
#include <string>
#include <vector>

class DataTypeAttribute : public Attribute
{
   public:
      DataTypeAttribute(std::unique_ptr<DataType>&& data, 
			AccessType accessType = AccessType::PUBLIC);
      DataTypeAttribute(const DataTypeAttribute& rv);
      void duplicate(std::unique_ptr<Attribute>&& dup) const;
      DataTypeAttribute& operator=(const DataTypeAttribute& rv);
      virtual ~DataTypeAttribute();
      
      const DataType* getDataType() const;
      void releaseDataType(std::unique_ptr<DataType>&& rv);
      void setDataType(std::unique_ptr<DataType>&& rv);

      virtual std::string getName() const;
      virtual std::string getType() const;
      virtual bool isBasic() const;
      virtual bool isPointer() const;
      virtual bool isOwned() const;

   private:
      void destructOwnedHeap();
      void copyOwnedHeap(const DataTypeAttribute& rv);
      DataType* _dataType;
};

#endif
