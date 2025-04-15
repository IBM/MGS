// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CUSTOMSTRINGDATAITEM_H
#define CUSTOMSTRINGDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include "CustomString.h"
#include <string>

class CustomStringDataItem : public DataItem
{
   private:
      std::string _data;

   public:
      virtual CustomStringDataItem& operator=(const CustomStringDataItem& DI);

      static const char* _type;

      // Constructors
      CustomStringDataItem(const std::string& data = "");
      CustomStringDataItem(CustomString& data);
      CustomStringDataItem(const CustomStringDataItem& DI);

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      std::string getString(Error* error=0) const;
      CustomString getLensString(Error* error=0) const;
      void setString(std::string i, Error* error=0);
};
#endif
