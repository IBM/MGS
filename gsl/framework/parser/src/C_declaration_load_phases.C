// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_load_phases.h"
#include "C_phase_list.h"
#include "DataItem.h"
#include "SyntaxError.h"
#include "C_production.h"
#include "LoadPhase.h"
#include "GslContext.h"

C_declaration* C_declaration_load_phases::duplicate() const
{
   return new C_declaration_load_phases(*this);
}

void C_declaration_load_phases::internalExecute(GslContext *c)
{
   std::unique_ptr<Phase> phase(new LoadPhase());
   c->setCurrentPhase(phase);
   _phaseList->execute(c);
}


C_declaration_load_phases::C_declaration_load_phases(
   const C_declaration_load_phases& rv)
   : C_declaration(rv), _phaseList(0)
{
   if (rv._phaseList) {
      _phaseList = rv._phaseList->duplicate();
   }
}


C_declaration_load_phases::C_declaration_load_phases(
   C_phase_list *a, SyntaxError * error)
   : C_declaration(error), _phaseList(a)
{
}

C_declaration_load_phases::~C_declaration_load_phases()
{
   delete _phaseList;
}

void C_declaration_load_phases::checkChildren() 
{
   if (_phaseList) {
      _phaseList->checkChildren();
      if (_phaseList->isError()) {
	 setError();
      }
   } 
} 

void C_declaration_load_phases::recursivePrint() 
{
   if (_phaseList) {
      _phaseList->recursivePrint();
   }
   printErrorMessage();
} 
