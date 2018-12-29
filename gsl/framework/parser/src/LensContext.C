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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "LensContext.h"
#include "Simulation.h"
#include "Repertoire.h"
#include "RepertoireDataItem.h"
#include "LensLexer.h"
#include "C_production.h"
#include "SyntaxErrorException.h"
#include "Phase.h"

#include <iostream>
#include <vector>

LensContext::LensContext(Simulation* s)
   : lexer(0), sim(s), symTable(), layerContext(0), 
     _error(false), _currentPhase(0)
{
  connectionContext = new ConnectionContext;
}

void LensContext::addCurrentRepertoire(Repertoire* rep)
{
   RepertoireDataItem* rdi = new RepertoireDataItem();
   rdi->setRepertoire(rep);
   std::unique_ptr<DataItem> ap_di(rdi);
   std::string currentRep("CurrentRepertoire");
   try {
      symTable.addEntry(currentRep, ap_di);
   } catch (SyntaxErrorException& e) {
      std::cerr << "While adding CurrentRepertoire, " 
		<< e.getError() << std::endl;
      throw;
   }
}


LensContext::LensContext(const LensContext& lc)
   : lexer(lc.lexer), sim(lc.sim), symTable(lc.symTable),
     connectionContext(0), layerContext(lc.layerContext),
     _error(lc._error), _currentPhase(0)
{
   if (lc._currentPhase) {
     std::unique_ptr<Phase> dup;
     lc._currentPhase->duplicate(dup);
     _currentPhase = dup.release();
   }

   std::vector<C_production*>::const_iterator it, end = lc._statements.end();
   for (it = lc._statements.begin(); it != end; ++it) {
      _statements.push_back((*it)->duplicate());
   }
   connectionContext = new ConnectionContext(*lc.connectionContext);
}

void LensContext::addStatement(C_production* statement)
{
   _statements.push_back(statement);
}

void LensContext::execute() 
{
   std::vector<C_production*>::iterator it, end = _statements.end();
#define DEBUG
#ifdef DEBUG
   int i = 1;
   sim->benchmark_set_timelapsed_diff();
#endif
   for (it = _statements.begin(); it != end; it++) {
      try {
	(*it)->execute(this);
#ifdef DEBUG 
	if (sim->getRank() == 0)
	  std::cout<< ".............statement " << i++ << std::endl;
	//sim->benchmark_timelapsed("... ");
	sim->benchmark_timelapsed_diff("... ");
#endif
      } catch (SyntaxErrorException& e) {
	e.printError();
	 (*it)->recursivePrint();
	 setError();	 
      } 
   }
}

LensContext::~LensContext()
{
   std::vector<C_production*>::iterator it, end = _statements.end();
   for (it = _statements.begin(); it != end; ++it) {
      delete *it;
   }
   delete _currentPhase;
   delete connectionContext;
}

void LensContext::getCurrentPhase(std::unique_ptr<Phase>& phase) const
{
   _currentPhase->duplicate(phase);
}

void LensContext::setCurrentPhase(std::unique_ptr<Phase>& phase)
{
   delete _currentPhase;
   _currentPhase = phase.release();
}

void LensContext::duplicate(std::unique_ptr<LensContext>& dup) const
{
   dup.reset(new LensContext(*this));
}

