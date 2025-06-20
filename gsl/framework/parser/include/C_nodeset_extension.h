// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_nodeset_extension_H
#define C_nodeset_extension_H
#include "Copyright.h"

#include <vector>
#include <list>
#include "C_production.h"

class C_node_type_set_specifier;
class C_index_set_specifier;
class GslContext;
class GridLayerDescriptor;
class Grid;
class SyntaxError;

class C_nodeset_extension : public C_production
{
   public:
      C_nodeset_extension(const C_nodeset_extension&);
      C_nodeset_extension(C_node_type_set_specifier *, 
			  C_index_set_specifier *, SyntaxError *);
      C_nodeset_extension(C_node_type_set_specifier *, SyntaxError *);
      C_nodeset_extension(C_index_set_specifier *, SyntaxError *);
      virtual ~C_nodeset_extension ();
      virtual C_nodeset_extension* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers(Grid* g);
      const std::vector<int>& getIndices();

   private:
      C_node_type_set_specifier* _nodeTypeSetSpecifier;
      C_index_set_specifier* _indexSetSpecifier;
      GslContext* _storedContext;
      std::list<GridLayerDescriptor*>* _layers;
      std::vector<int> _indices;
      Grid* _lastGrid;
};
#endif
