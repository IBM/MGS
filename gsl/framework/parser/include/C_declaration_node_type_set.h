// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_node_type_set_H
#define C_declaration_node_type_set_H
#include "Copyright.h"
#include "C_declaration.h"

class C_declarator;
class C_node_type_set_specifier_clause;
class GslContext;
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
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_node_type_set_specifier_clause* _nodeTypeSetSpecifierClause;
};
#endif
