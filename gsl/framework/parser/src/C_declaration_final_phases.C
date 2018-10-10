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

#include "C_declaration_final_phases.h"
#include "C_phase_list.h"
#include "DataItem.h"
#include "SyntaxError.h"
#include "C_production.h"
#include "FinalPhase.h"
#include "LensContext.h"

C_declaration* C_declaration_final_phases::duplicate() const
{
   return new C_declaration_final_phases(*this);
}

void C_declaration_final_phases::internalExecute(LensContext *c)
{
   std::unique_ptr<Phase> phase(new FinalPhase());
   c->setCurrentPhase(phase);
   _phaseList->execute(c);
}


C_declaration_final_phases::C_declaration_final_phases(
   const C_declaration_final_phases& rv)
   : C_declaration(rv), _phaseList(0)
{
   if (rv._phaseList) {
      _phaseList = rv._phaseList->duplicate();
   }
}


C_declaration_final_phases::C_declaration_final_phases(
   C_phase_list *a, SyntaxError * error)
   : C_declaration(error), _phaseList(a)
{
}

C_declaration_final_phases::~C_declaration_final_phases()
{
   delete _phaseList;
}

void C_declaration_final_phases::checkChildren() 
{
   if (_phaseList) {
      _phaseList->checkChildren();
      if (_phaseList->isError()) {
	 setError();
      }
   } 
} 

void C_declaration_final_phases::recursivePrint() 
{
   if (_phaseList) {
      _phaseList->recursivePrint();
   }
   printErrorMessage();
} 
