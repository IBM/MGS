// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NODETYPESETDATAITEM_H
#define NODETYPESETDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class NodeTypeSet;

class NodeTypeSetDataItem : public DataItem
{
   private:
      NodeTypeSet *_nodeTypeSet;

   public:
      static const char* _type;

      virtual NodeTypeSetDataItem& operator=(const NodeTypeSetDataItem& DI);

      // Constructors
      NodeTypeSetDataItem();
      NodeTypeSetDataItem(std::unique_ptr<NodeTypeSet> nodeTypeSet);
      NodeTypeSetDataItem(const NodeTypeSetDataItem& DI);

      // Destructor
      ~NodeTypeSetDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      NodeTypeSet* getNodeTypeSet() const;
      void setNodeTypeSet(NodeTypeSet* nts);

};
#endif
