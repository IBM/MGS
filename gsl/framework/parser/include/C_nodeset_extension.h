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

#ifndef C_nodeset_extension_H
#define C_nodeset_extension_H
#include "Copyright.h"

#include <vector>
#include <list>
#include "C_production.h"

class C_node_type_set_specifier;
class C_index_set_specifier;
class LensContext;
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
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers(Grid* g);
      const std::vector<int>& getIndices();

   private:
      C_node_type_set_specifier* _nodeTypeSetSpecifier;
      C_index_set_specifier* _indexSetSpecifier;
      LensContext* _storedContext;
      std::list<GridLayerDescriptor*>* _layers;
      std::vector<int> _indices;
      Grid* _lastGrid;
};
#endif
