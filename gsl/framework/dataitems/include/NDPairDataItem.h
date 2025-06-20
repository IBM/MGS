// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NDPAIRDATAITEM_H
#define NDPAIRDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <memory>

class NDPair;

class NDPairDataItem : public DataItem
{
   public:
      static char const* _type;

      NDPairDataItem & operator=(const NDPairDataItem &DI);

      // Constructors
      NDPairDataItem();
      NDPairDataItem(std::unique_ptr<NDPair> data);
      NDPairDataItem(const NDPairDataItem& DI);

      // Destructor
      ~NDPairDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      NDPair* getNDPair() const;
      void releaseNDPair(std::unique_ptr<NDPair>& ndp);
      void setNDPair(std::unique_ptr<NDPair>& ndp);
   private:
      NDPair *_data;
      inline void copyContents(const NDPairDataItem& DI);
      inline void destructContents();
};
#endif
