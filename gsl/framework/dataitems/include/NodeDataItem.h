// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NODEDATAITEM_H
#define NODEDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class Node;

class NodeDataItem : public DataItem
{
   private:
      Node *_node;

   public:
      static const char* _type;

      virtual NodeDataItem& operator=(const NodeDataItem& DI);

      // Constructors
      NodeDataItem(Node *node = 0);
      NodeDataItem(const NodeDataItem& DI);

      // Destructor
      ~NodeDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Node* getNode() const;
      void setNode(Node* ns);
      std::string getString(Error* error=0) const;

};
#endif
