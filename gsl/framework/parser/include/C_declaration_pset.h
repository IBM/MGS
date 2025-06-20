// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_pset_H
#define C_declaration_pset_H
#include "Copyright.h"

#include "C_declaration.h"
#include "ParameterSet.h"

class C_parameter_type_pair;
class C_declarator;
class C_ndpair_clause_list;
class GslContext;

class C_declaration_pset : public C_declaration
{
   public:
      C_declaration_pset(const C_declaration_pset&);
      C_declaration_pset(C_parameter_type_pair *, C_declarator *, 
			 C_ndpair_clause_list *, SyntaxError *);
      virtual C_declaration_pset* duplicate() const;
      virtual ~C_declaration_pset();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_parameter_type_pair* _parameterTypePair;
      C_declarator* _declarator;
      C_ndpair_clause_list* _ndpClauseList;
};
#endif
