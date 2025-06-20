// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NODESETDATAITEM_H
#define NODESETDATAITEM_H
#include "Copyright.h"

#include "TriggerableDataItem.h"
#include <vector>

class NodeSet;
class GridSet;
class Triggerable;

class NodeSetDataItem : public TriggerableDataItem
{
   private:
      NodeSet *_nodeset;

   public:
      static const char* _type;

      virtual NodeSetDataItem& operator=(const NodeSetDataItem& DI);

      // Constructors
      NodeSetDataItem();
      NodeSetDataItem(std::unique_ptr<NodeSet>& nodeset);
      NodeSetDataItem(const NodeSetDataItem& DI);

      // Destructor
      ~NodeSetDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      NodeSet* getNodeSet() const;
      void setNodeSet(NodeSet* ns);
      void setNodeSet(GridSet* ns);
      std::string getString(Error* error=0) const;

      virtual std::vector<Triggerable*> getTriggerables();
};
#endif
