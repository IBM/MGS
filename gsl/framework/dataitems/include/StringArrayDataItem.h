// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef STRINGARRAYDATAITEM_H
#define STRINGARRAYDATAITEM_H
#include "Copyright.h"

#include "ArrayDataItem.h"
#include "ShallowArray.h"

class StringArrayDataItem : public ArrayDataItem
{
   private:
      StringArrayDataItem & operator=(StringArrayDataItem const &);
      StringArrayDataItem & assign(const StringArrayDataItem &);
      std::vector<std::string> *_data;

   public:
      static const char* _type;

      // Constructors
      StringArrayDataItem();
      StringArrayDataItem(const StringArrayDataItem& DI);
      StringArrayDataItem(std::vector<int> const & dimensions);
      StringArrayDataItem(ShallowArray<std::string> const & data);
      ~StringArrayDataItem();

      // Utility methods
      void setDimensions(std::vector<int> const &dimensions);
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Array Methods
      std::string getString(Error* error=0) const;
      std::string getString(std::vector<int> coords, Error* error=0) const;
      void setString(std::vector<int> coords, std::string value, Error* error=0);
      const std::vector<std::string>* getStringVector() const;
      std::vector<std::string>* getModifiableStringVector();
};
#endif
