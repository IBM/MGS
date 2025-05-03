// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_TRIGGER_SPECIFIER_H
#define C_TRIGGER_SPECIFIER_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class C_declarator;
class GslContext;
class C_trigger;
class SyntaxError;
class C_ndpair_clause_list;

class C_trigger_specifier : public C_production
{
   public:
      C_trigger_specifier(const C_trigger_specifier&);
      C_trigger_specifier(C_declarator *, C_declarator *, 
			  C_ndpair_clause_list *, C_trigger *, SyntaxError *);
      C_trigger_specifier(std::string, C_trigger *, SyntaxError *);
      virtual ~C_trigger_specifier();
      virtual C_trigger_specifier* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _triggerable;
      C_declarator* _action;
      C_trigger* _trigger;
      C_ndpair_clause_list* _ndpairList;
      std::string _triggerableSpecifier;
};
#endif
