// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Node* getFirstNode() const;
      Node* getSecondNode() const;
      void setFirstNode(Node* ns);
      void setSecondNode(Node* ns);
      std::string getString(Error* error=0) const;

};
#endif
