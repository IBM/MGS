// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_ndpairlist_H
#define C_declaration_ndpairlist_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class C_ndpair_clause_list;
class LensContext;
class SyntaxError;

class C_declaration_ndpairlist : public C_declaration
{
   public:
      C_declaration_ndpairlist(const C_declaration_ndpairlist&);
      C_declaration_ndpairlist(C_declarator *, C_ndpair_clause_list *, 
			       SyntaxError *);
      virtual C_declaration_ndpairlist* duplicate() const;
      virtual ~C_declaration_ndpairlist();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_ndpair_clause_list* _ndp_clause_list;
};
#endif
