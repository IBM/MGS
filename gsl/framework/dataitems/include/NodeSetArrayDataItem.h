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

#ifndef NODESETARRAYDATAITEM_H
#define NODESETARRAYDATAITEM_H
#include "Copyright.h"

#include "ArrayDataItem.h"
#include "ShallowArray.h"

class NodeSetArrayDataItem : public ArrayDataItem
{
   protected:
      NodeSetArrayDataItem & operator=(NodeSetArrayDataItem const &);
      NodeSetArrayDataItem & assign(const NodeSetArrayDataItem &);

   private:
      std::vector<NodeSet*> *_data;

   public:
      static const char* _type;

      // Constructors
      NodeSetArrayDataItem();
      NodeSetArrayDataItem(const NodeSetArrayDataItem& DI);
      NodeSetArrayDataItem(ShallowArray<NodeSet*> const & data);
      NodeSetArrayDataItem(std::vector<int> const & dimensions);

      // Destructor
      ~NodeSetArrayDataItem();

      // Utility methods
      void setDimensions(std::vector<int> const &dimensions);
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Array Methods
      NodeSet* getNodeSet(std::vector<int> coords, Error* error=0) const;
      void setNodeSet(std::vector<int> coords, NodeSet* value, Error* error=0);
      const std::vector<NodeSet*>* getNodeSetVector() const;
      std::vector<NodeSet*>* getModifiableNodeSetVector();
};
#endif
