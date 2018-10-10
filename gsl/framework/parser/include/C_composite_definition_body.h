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
