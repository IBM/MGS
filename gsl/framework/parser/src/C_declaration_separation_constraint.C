// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_separation_constraint.h"
#include "DataItem.h"
#include "SyntaxError.h"
#include "C_production.h"
#include "LensContext.h"
#include "C_separation_constraint.h"
#include "C_separation_constraint_list.h"
#include "Simulation.h"
#include "GranuleSet.h"

void C_declaration_separation_constraint::internalExecute(LensContext *c)
{
   if (c->sim->isCostAggregationPass()) {
      _separationConstraintList->execute(c);
      
      const std::vector<C_separation_constraint*>& separationConstraints = 
	 _separationConstraintList->getList();
      
      std::vector<C_separation_constraint*>::const_iterator it, 
	 end = separationConstraints.end();
      
      GranuleSet granuleSet;
      for (it = separationConstraints.begin(); it != end; ++it) {
	 if ((*it)->getVariableConstraint()) {
	    granuleSet.insert(
	       c->sim->getGranule(*(*it)->getVariableConstraint()));
	 } else if ((*it)->getNodeSetConstraint()) {
	    c->sim->getGranules(*(*it)->getNodeSetConstraint(), granuleSet);
	 } else {
	    throwError("Internal error: C_declaration_separation_constraint");
	 }
      }
      c->sim->addUnseparableGranuleSet(granuleSet);
   }
}


C_declaration_separation_constraint::C_declaration_separation_constraint(
   const C_declaration_separation_constraint& rv)
   : C_declaration(rv), _separationConstraintList(0)
{
   if (rv._separationConstraintList) {
      _separationConstraintList = rv._separationConstraintList->duplicate();
   }
}


C_declaration_separation_constraint::C_declaration_separation_constraint(
   C_separation_constraint_list *a, SyntaxError * error)
   : C_declaration(error), _separationConstraintList(a)
{
}

C_declaration_separation_constraint::~C_declaration_separation_constraint()
{
   delete _separationConstraintList;
}

C_declaration* C_declaration_separation_constraint::duplicate() const
{
   return new C_declaration_separation_constraint(*this);
}

void C_declaration_separation_constraint::checkChildren() 
{
   if (_separationConstraintList) {
      _separationConstraintList->checkChildren();
      if (_separationConstraintList->isError()) {
	 setError();
      }
   } 
} 

void C_declaration_separation_constraint::recursivePrint() 
{
   if (_separationConstraintList) {
      _separationConstraintList->recursivePrint();
   }
   printErrorMessage();
} 
