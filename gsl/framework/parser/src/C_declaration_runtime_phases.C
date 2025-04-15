// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_runtime_phases.h"
#include "C_phase_list.h"
#include "DataItem.h"
#include "SyntaxError.h"
#include "C_production.h"
#include "RuntimePhase.h"
#include "LensContext.h"

C_declaration* C_declaration_runtime_phases::duplicate() const
{
   return new C_declaration_runtime_phases(*this);
}

void C_declaration_runtime_phases::internalExecute(LensContext *c)
{
   std::unique_ptr<Phase> phase(new RuntimePhase());
   c->setCurrentPhase(phase);
   _phaseList->execute(c);
}


C_declaration_runtime_phases::C_declaration_runtime_phases(
   const C_declaration_runtime_phases& rv)
   : C_declaration(rv), _phaseList(0)
{
   if (rv._phaseList) {
      _phaseList = rv._phaseList->duplicate();
   }
}


C_declaration_runtime_phases::C_declaration_runtime_phases(
   C_phase_list *a, SyntaxError * error)
   : C_declaration(error), _phaseList(a)   
{
}

C_declaration_runtime_phases::~C_declaration_runtime_phases()
{
   delete _phaseList;
}

void C_declaration_runtime_phases::checkChildren() 
{
   if (_phaseList) {
      _phaseList->checkChildren();
      if (_phaseList->isError()) {
	 setError();
      }
   } 
} 

void C_declaration_runtime_phases::recursivePrint() 
{
   if (_phaseList) {
      _phaseList->recursivePrint();
   }
   printErrorMessage();
} 
