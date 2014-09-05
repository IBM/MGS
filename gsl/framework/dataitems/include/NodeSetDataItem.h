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
      NodeSetDataItem(std::auto_ptr<NodeSet> nodeset);
      NodeSetDataItem(const NodeSetDataItem& DI);

      // Destructor
      ~NodeSetDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      NodeSet* getNodeSet() const;
      void setNodeSet(NodeSet* ns);
      void setNodeSet(GridSet* ns);
      std::string getString(Error* error=0) const;

      virtual std::vector<Triggerable*> getTriggerables();
};
#endif
