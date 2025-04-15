#include <memory>
// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "DataTypeAttribute.h"
#include "AccessType.h"
#include "DataType.h"
#include <string>
#include <vector>

DataTypeAttribute::DataTypeAttribute(std::unique_ptr<DataType>&& data,
				     AccessType accessType)
   : Attribute(accessType)
{
   _dataType = data.release();
}

DataTypeAttribute::DataTypeAttribute(const DataTypeAttribute& rv)
   : Attribute(rv), _dataType(0)
{
   copyOwnedHeap(rv);
}

void DataTypeAttribute::duplicate(std::unique_ptr<Attribute>&& dup)const
{
   dup.reset(new DataTypeAttribute(*this));
}

DataTypeAttribute& DataTypeAttribute::operator=(const DataTypeAttribute& rv)
{
   if (this != &rv) {
      Attribute::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}


DataTypeAttribute::~DataTypeAttribute()
{
   destructOwnedHeap();
}

std::string DataTypeAttribute::getName() const
{
   return _dataType->getName();
}

std::string DataTypeAttribute::getType() const
{
   return _dataType->getDescriptor();
}

bool DataTypeAttribute::isBasic() const
{
   return _dataType->isBasic();
}

bool DataTypeAttribute::isPointer() const
{
   return _dataType->isPointer();
}

bool DataTypeAttribute::isOwned() const
{
   return _dataType->shouldBeOwned();
}

const DataType* DataTypeAttribute::getDataType() const
{
   return _dataType;
}

void DataTypeAttribute::releaseDataType(std::unique_ptr<DataType>&& rv)
{
   rv.reset(_dataType);
   _dataType = 0;
}

void DataTypeAttribute::setDataType(std::unique_ptr<DataType>&& rv)
{
   delete _dataType;
   _dataType = rv.release();
}

void DataTypeAttribute::destructOwnedHeap()
{
   delete _dataType;
}

void DataTypeAttribute::copyOwnedHeap(const DataTypeAttribute& rv)
{   
   if (rv._dataType) {
      std::unique_ptr<DataType> dup;
      rv._dataType->duplicate(std::move(dup));
      _dataType = dup.release();
   } else {
      _dataType = 0;
   }
}
