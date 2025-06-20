// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NDPAIRLISTDATAITEM_H
#define NDPAIRLISTDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include "NDPairList.h"

#include <list>
#include <memory>

class NDPairListDataItem : public DataItem
{
   public:
      static const char* _type;

      NDPairListDataItem & operator=(const NDPairListDataItem & DI);

      // Constructors
      NDPairListDataItem();
      NDPairListDataItem(std::unique_ptr<NDPairList>& data);
      NDPairListDataItem(const NDPairListDataItem& DI);
      
      // Destructor
      ~NDPairListDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      NDPairList* getNDPairList(Error* error=0) const;
      void releaseNDPairList(std::unique_ptr<NDPairList>& ap, Error* error=0);
      void setNDPairList(std::unique_ptr<NDPairList>& ap, Error* error=0);
      virtual std::string getString(Error* error=0) const;
   private:
      NDPairList* _data;
      inline void copyContents(const NDPairListDataItem& DI);
      inline void destructContents();
};
#endif
