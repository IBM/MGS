// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef STRUCTDATAITEM_H
#define STRUCTDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <vector>
#include <memory>

class Struct;

class StructDataItem : public DataItem
{

   private:
      Struct *_data;
      void copyOwnedHeap(const StructDataItem& rv);
      void destructOwnedHeap();

   public:
      static char const* _type;

      StructDataItem & operator=(const StructDataItem &);

      // Constructors
      StructDataItem();
      StructDataItem(std::unique_ptr<Struct>& data);
      StructDataItem(const StructDataItem& rv);

      // Destructor
      ~StructDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      Struct* getStruct(Error* error=0) const;
      void setStruct(std::unique_ptr<Struct>& s, Error* error=0);
      std::string getString(Error* error=0) const;

};
#endif
