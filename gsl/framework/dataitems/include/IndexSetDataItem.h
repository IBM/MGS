// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef INDEXSETDATAITEM_H
#define INDEXSETDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

#include "IndexSet.h"

class IndexSetDataItem : public DataItem
{
   private:
      IndexSet _set;

   public:
      static char const* _type;

      IndexSetDataItem & operator=(const IndexSetDataItem &);

      // Constructors
      IndexSetDataItem();
      IndexSetDataItem(const IndexSetDataItem& DI);

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      void setIndexSet(IndexSet*);
      const IndexSet* getIndexSet() const;
};
#endif
