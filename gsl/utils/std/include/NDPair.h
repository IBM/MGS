// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NDPAIR_H
#define NDPAIR_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "DataItem.h"

class NDPair
{
   public:
      NDPair(const std::string& name, std::unique_ptr<DataItem>& di);
      NDPair(const std::string& name, const std::string& value);
      NDPair(const std::string& name, double value);
      NDPair(const std::string& name, int value);
      NDPair(const NDPair& rv);
      const NDPair& operator=(const NDPair& rv);
      const std::string& getName() const;
      std::string getValue() const;
      DataItem* getDataItem() const;
      void setDataItem(DataItem* di);
      void setValue(const std::string& value);
      void setValue(int value);
      void setValue(double value);
      void getDataItemOwnership(std::unique_ptr<DataItem>& di);
      void setDataItemOwnership(std::unique_ptr<DataItem>& di);
      int operator==(const std::string& n) const;
      int operator!=(const std::string& n) const;
      ~NDPair();
   private:
      void copyContents(const NDPair& rv);
      void destructContents();
      std::string _name;
      std::unique_ptr<DataItem> _dataItem;
};
#endif
