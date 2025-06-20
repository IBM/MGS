// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Array Methods
      NodeSet* getNodeSet(std::vector<int> coords, Error* error=0) const;
      void setNodeSet(std::vector<int> coords, NodeSet* value, Error* error=0);
      const std::vector<NodeSet*>* getNodeSetVector() const;
      std::vector<NodeSet*>* getModifiableNodeSetVector();
};
#endif
