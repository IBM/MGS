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

#include "C_separation_constraint.h"

#include "LensContext.h"
#include "SyntaxError.h"
#include "C_declarator.h"
#include "C_nodeset.h"
#include "NodeSet.h"
#include "NodeSetDataItem.h"
#include "Variable.h"
#include "VariableInstanceAccessor.h"
#include "VariableDataItem.h"


void C_separation_constraint::internalExecute(LensContext *c)
{
   if (_declarator) {
      _declarator->execute(c);

      DataItem* di = c->symTable.getEntry(_declarator->getName());
      if (!di) {
	 std::string mes = "Argument " + _declarator->getName() + 
	    " was not declared";
	 throwError(mes);
      }

      NodeSetDataItem* nsdi = dynamic_cast<NodeSetDataItem*>(di);
      if (nsdi) {
	 _nodesetConstraint = nsdi->getNodeSet();
      } else {
	 VariableDataItem* vdi = dynamic_cast<VariableDataItem*>(di);
	 if (vdi) {
	    _variableConstraint = vdi->getVariable()->getVariable();    // modified by Jizhu Lu on 03/16/2006
	 } else {
	    std::string mes = "Argument " + _declarator->getName() + 
	       " is not a NodeSet or Variable.";
	    throwError(mes);
	 }
      }
   } else if (_nodeset) {
      _nodeset->execute(c);
      _nodesetConstraint = _nodeset->getNodeSet();
   } else {
      throwError("Internal error: Separation constraint");
   }
}

C_separation_constraint::C_separation_constraint(
   C_declarator *declarator, SyntaxError* error)
   : C_production(error), _declarator(declarator), _nodeset(0), 
     _nodesetConstraint(0), _variableConstraint(0)
{ 
}

C_separation_constraint::C_separation_constraint(
   C_nodeset* nodeset, SyntaxError* error)
   : C_production(error), _declarator(0), _nodeset(nodeset),
     _nodesetConstraint(0), _variableConstraint(0)
   
{
}

C_separation_constraint::C_separation_constraint(
   const C_separation_constraint& rv)
   : C_production(rv), _declarator(0), _nodeset(0), 
     _nodesetConstraint(rv._nodesetConstraint), 
     _variableConstraint(rv._variableConstraint)

{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._nodeset) {
      _nodeset = rv._nodeset->duplicate();
   }
}

C_separation_constraint::~C_separation_constraint()
{
   delete _declarator;
   delete _nodeset;
}

C_separation_constraint* C_separation_constraint::duplicate() const
{
   return new C_separation_constraint(*this);
}

void C_separation_constraint::checkChildren()
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_nodeset) {
      _nodeset->checkChildren();
      if (_nodeset->isError()) {
         setError();
      }
   }
}

void C_separation_constraint::recursivePrint()
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_nodeset) {
      _nodeset->recursivePrint();
   }
   printErrorMessage();
}
