// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GSLCONTEXT_H
#define GSLCONTEXT_H
#include "Copyright.h"

#include "SymbolTable.h"
#include "ConnectionContext.h"

#include <string>
#include <vector>
#include <memory>

class Repertoire;
class GslLexer;
class Simulation;
class LayerDefinitionContext;
class C_production;
class Phase;
class CompCategoryBase;

class GslContext
{
   public:
      // constructors
      GslContext(Simulation* s);
      GslContext(const GslContext& lc);

      // members
      GslLexer *lexer;
      Simulation* sim;
      SymbolTable symTable;
      ConnectionContext* connectionContext;
      LayerDefinitionContext *layerContext;

      void setError() { _error = true;};
      bool isError() { return _error;};
      void addStatement(C_production* statement);
      void execute();
      
      void getCurrentPhase(std::unique_ptr<Phase>& phase) const;
      void setCurrentPhase(std::unique_ptr<Phase>& phase);

      CompCategoryBase* getCurrentCompCategoryBase() const {
	 return _currentCompCategoryBase;
      }
      void setCurrentCompCategoryBase(CompCategoryBase* cc) {
	 _currentCompCategoryBase = cc;
      }
      
      virtual void duplicate(std::unique_ptr<GslContext>& dup) const;

      void addCurrentRepertoire(Repertoire* rep);

      // destructor
      virtual ~GslContext();
   private:
      bool _error;
      std::vector<C_production*> _statements;
      Phase* _currentPhase;
      CompCategoryBase* _currentCompCategoryBase;
};
#endif
