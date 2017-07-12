// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      NodeTypeSetDataItem(std::auto_ptr<NodeTypeSet> nodeTypeSet);
      NodeTypeSetDataItem(const NodeTypeSetDataItem& DI);

      // Destructor
      ~NodeTypeSetDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      NodeTypeSet* getNodeTypeSet() const;
      void setNodeTypeSet(NodeTypeSet* nts);

};
#endif
