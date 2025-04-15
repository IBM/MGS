// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_composite_definition_body_H
#define C_composite_definition_body_H
#include "Copyright.h"

#include "RepertoireFactory.h"
#include "C_production.h"
#include <string>

class C_composite_statement_list;
class LensContext;
class Repertoire;
class SyntaxError;

class C_composite_definition_body : public C_production, public RepertoireFactory
{
   public:
      C_composite_definition_body(const C_composite_definition_body&);
      C_composite_definition_body(C_composite_statement_list *, SyntaxError *);
      virtual ~C_composite_definition_body();
      virtual C_composite_definition_body* duplicate() const;
      virtual void duplicate(std::unique_ptr<RepertoireFactory>& rv) const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      Repertoire* createRepertoire(const std::string& repName, LensContext* c);
      void setTdError(SyntaxError *tdError);
      
   private:
      C_composite_statement_list* _statementList;
};
#endif
