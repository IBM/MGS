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

#ifndef C_node_type_set_specifier_H
#define C_node_type_set_specifier_H
#include "Copyright.h"

#include <list>
#include "C_production_grid.h"

class C_node_type_set_specifier_clause;
class Grid;
class GridLayerDescriptor;
class LensContext;
class SyntaxError;

class C_node_type_set_specifier : public C_production_grid
{
   public:
      C_node_type_set_specifier(const C_node_type_set_specifier&);
      C_node_type_set_specifier(C_node_type_set_specifier_clause *, 
				SyntaxError *);
      virtual ~C_node_type_set_specifier ();
      virtual C_node_type_set_specifier* duplicate() const;
      virtual void internalExecute(LensContext *, Grid* g);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<GridLayerDescriptor*>& getLayers() const;

   private:
      C_node_type_set_specifier_clause* _nodeTypeSetSpecifierClause;
      std::list<GridLayerDescriptor*> _layers;
};
#endif
