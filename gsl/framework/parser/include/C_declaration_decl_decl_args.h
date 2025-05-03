// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_decl_decl_args_H
#define C_declaration_decl_decl_args_H
#include "Copyright.h"

#include "C_declaration.h"
#include <string>
class C_declarator;
class C_argument_list;
class C_ndpair_clause_list;
class GslContext;
class InstanceFactory;
class DataItem;
class SyntaxError;

class C_declaration_decl_decl_args : public C_declaration
{
   public:
      C_declaration_decl_decl_args(const C_declaration_decl_decl_args&);
      C_declaration_decl_decl_args(C_declarator *, C_declarator *, 
				   C_argument_list *, SyntaxError *);
      C_declaration_decl_decl_args(C_ndpair_clause_list *, C_declarator *, 
				   C_declarator *, SyntaxError *);
      virtual ~C_declaration_decl_decl_args();
      virtual C_declaration_decl_decl_args* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::string getName();

   private:
      C_declarator* _userDefinedType;
      C_declarator* _nameDecl;
      C_argument_list* _argList;
      C_ndpair_clause_list* _ndpair_clause_list;
      DataItem* _dataitem;
};
#endif
