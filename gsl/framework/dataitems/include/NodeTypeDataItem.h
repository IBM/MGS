// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NODETYPEDATAITEM_H
#define NODETYPEDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

#include <memory>

class NodeType;

class NodeTypeDataItem : public DataItem
{
   private:
      NodeType *_data;

   public:
      static const char* _type;

      virtual NodeTypeDataItem& operator=(const NodeTypeDataItem& DI);

      // Constructors
      NodeTypeDataItem(NodeType *data = 0);
      NodeTypeDataItem(const NodeTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;

      void setNodeType(NodeType*);
      NodeType* getNodeType() const;
};
#endif
