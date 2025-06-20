// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CONSTANTDATAITEM_H
#define CONSTANTDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <vector>
#include <memory>

class Constant;

class ConstantDataItem : public DataItem
{

   private:
      Constant *_data;
      void copyOwnedHeap(const ConstantDataItem& rv);
      void destructOwnedHeap();

   public:
      static char const* _type;

      ConstantDataItem& operator=(const ConstantDataItem& rv);

      // Constructors
      ConstantDataItem();
      ConstantDataItem(std::unique_ptr<Constant> data);
      ConstantDataItem(const ConstantDataItem& rv);

      // Destructor
      ~ConstantDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      Constant* getConstant(Error* error=0) const;
      void setConstant(std::unique_ptr<Constant>& c, Error* error=0);
      std::string getString(Error* error=0) const;

};
#endif
