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

#ifndef C_grid_definition_body_H
#define C_grid_definition_body_H
#include "Copyright.h"

#include "RepertoireFactory.h"
#include "C_production.h"

class C_dim_declaration;
class C_grid_translation_unit;
class LensContext;
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
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      Repertoire* createRepertoire(std::string const& repName, LensContext* c);
      void setTdError(SyntaxError *tdError);

   private:
      C_dim_declaration* _dimDecl;
      C_grid_translation_unit* _gridTransUnit;

};
#endif
