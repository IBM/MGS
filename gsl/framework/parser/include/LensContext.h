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

#ifndef LENSCONTEXT_H
#define LENSCONTEXT_H
#include "Copyright.h"

#include "SymbolTable.h"
#include "ConnectionContext.h"

#include <string>
#include <vector>
#include <memory>

class Repertoire;
class LensLexer;
class Simulation;
class LensDeletable;
class LayerDefinitionContext;
class C_production;
class Phase;
class CompCategoryBase;

class LensContext
{
   public:
      // constructors
      LensContext(Simulation* s);
      LensContext(const LensContext& lc);

      // members
      LensLexer *lexer;
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
      
      virtual void duplicate(std::unique_ptr<LensContext>& dup) const;

      void addCurrentRepertoire(Repertoire* rep);

      // destructor
      virtual ~LensContext();
   private:
      bool _error;
      std::vector<C_production*> _statements;
      Phase* _currentPhase;
      CompCategoryBase* _currentCompCategoryBase;
};
#endif
