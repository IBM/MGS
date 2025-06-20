// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_rel_nodeset_H
#define C_declaration_rel_nodeset_H
#include "Copyright.h"

#include "C_declaration.h"
class C_declarator;
class C_relative_nodeset;
class GslContext;
class SyntaxError;

class C_declaration_rel_nodeset : public C_declaration
{
   public:
      C_declaration_rel_nodeset(const C_declaration_rel_nodeset&);
      C_declaration_rel_nodeset(C_declarator *, C_relative_nodeset *, 
				SyntaxError *);
      virtual C_declaration_rel_nodeset* duplicate() const;
      virtual ~C_declaration_rel_nodeset();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declaration;
      C_relative_nodeset* _relativeNodeSet;

};
#endif
