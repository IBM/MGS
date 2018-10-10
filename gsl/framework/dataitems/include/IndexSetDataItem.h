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
