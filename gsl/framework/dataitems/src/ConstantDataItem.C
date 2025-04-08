// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "ConstantDataItem.h"
#include "Constant.h"

// Type
const char* ConstantDataItem::_type = "CONSTANT";

// Constructors
ConstantDataItem::ConstantDataItem()
   : _data(0)
{
}

ConstantDataItem::ConstantDataItem(std::unique_ptr<Constant> data)
{
   _data = data.release();
}

ConstantDataItem::~ConstantDataItem()
{
   destructOwnedHeap();
}

ConstantDataItem::ConstantDataItem(const ConstantDataItem& rv)
   : DataItem(rv), _data(0)
{
   copyOwnedHeap(rv);
}

// Utility methods
void ConstantDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new ConstantDataItem(*this));
}

ConstantDataItem& ConstantDataItem::operator=(const ConstantDataItem& rv)
{
   if (this != &rv) {
      DataItem::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

const char* ConstantDataItem::getType() const
{
   return _type;
}

// Singlet methods
Constant* ConstantDataItem::getConstant(Error* error) const
{
   return _data;
}

void ConstantDataItem::setConstant(std::unique_ptr<Constant>& c , Error* error)
{
   delete _data;
   _data = c.release();
}

std::string ConstantDataItem::getString(Error* error) const
{
   return "";
}

void ConstantDataItem::copyOwnedHeap(const ConstantDataItem& rv)
{
   if (rv._data) {
      std::unique_ptr<Constant> dup;
      rv._data->duplicate(std::move(dup));
      _data = dup.release();
   } else {
      _data = 0;
   }
}

void ConstantDataItem::destructOwnedHeap()
{
   delete _data;
}
