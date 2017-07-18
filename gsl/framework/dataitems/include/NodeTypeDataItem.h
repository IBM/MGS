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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;

      void setNodeType(NodeType*);
      NodeType* getNodeType() const;
};
#endif
