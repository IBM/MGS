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

#ifndef C_declaration_node_type_set_H
#define C_declaration_node_type_set_H
#include "Copyright.h"
#include "C_declaration.h"

class C_declarator;
class C_node_type_set_specifier_clause;
class LensContext;
class Grid;
class SyntaxError;

class C_declaration_node_type_set : public C_declaration
{
   public:
      C_declaration_node_type_set(const C_declaration_node_type_set&);
      C_declaration_node_type_set(
	 C_declarator *, C_node_type_set_specifier_clause *, SyntaxError *);
      virtual C_declaration_node_type_set* duplicate() const;
      virtual ~C_declaration_node_type_set();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_node_type_set_specifier_clause* _nodeTypeSetSpecifierClause;
};
#endif
