// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NODEPAIRDATAITEM_H
#define NODEPAIRDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class Node;

class NodePairDataItem : public DataItem
{
   private:
      Node *_first;
      Node *_second;

   public:
      static const char* _type;

      virtual NodePairDataItem& operator=(const NodePairDataItem& DI);

      // Constructors
      NodePairDataItem();
      NodePairDataItem(const NodePairDataItem& DI);

      // Destructor
      ~NodePairDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Node* getFirstNode() const;
      Node* getSecondNode() const;
      void setFirstNode(Node* ns);
      void setSecondNode(Node* ns);
      std::string getString(Error* error=0) const;

};
#endif
