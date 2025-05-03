// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_grid_definition_body_H
#define C_grid_definition_body_H
#include "Copyright.h"

#include "RepertoireFactory.h"
#include "C_production.h"

class C_dim_declaration;
class C_grid_translation_unit;
class GslContext;
class C_declarator;
class Repertoire;
class SyntaxError;

class C_grid_definition_body : public C_production, public RepertoireFactory
{
   public:
      C_grid_definition_body(const C_grid_definition_body&);
      C_grid_definition_body(C_dim_declaration *, SyntaxError *);
      C_grid_definition_body(C_dim_declaration *, C_grid_translation_unit *, 
			     SyntaxError *);
      virtual ~C_grid_definition_body();
      virtual C_grid_definition_body* duplicate() const;
      virtual void duplicate(std::unique_ptr<RepertoireFactory>& rv) const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      Repertoire* createRepertoire(std::string const& repName, GslContext* c);
      void setTdError(SyntaxError *tdError);

   private:
      C_dim_declaration* _dimDecl;
      C_grid_translation_unit* _gridTransUnit;

};
#endif
