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
