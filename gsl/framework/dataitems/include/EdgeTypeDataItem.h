// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EDGETYPEDATAITEM_H
#define EDGETYPEDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class EdgeType;

class EdgeTypeDataItem : public DataItem
{
   protected:
      DataItem & assign(const DataItem &);

   private:
      EdgeType *_data;

   public:
      static char const* _type;

      EdgeTypeDataItem(EdgeType *data = 0);
      EdgeTypeDataItem (const EdgeTypeDataItem& DI);

      virtual EdgeTypeDataItem& operator=(const EdgeTypeDataItem& DI);

      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      void setEdgeType(EdgeType*);
      EdgeType* getEdgeType() const;

};
#endif
