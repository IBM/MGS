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

#include "C_composite_definition_body.h"
#include "LensContext.h"
#include "C_composite_statement_list.h"
#include "Repertoire.h"

#include "RepertoireDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

void C_composite_definition_body::internalExecute(LensContext *c)
{
   //_statementList->execute(c);
}


C_composite_definition_body::C_composite_definition_body(
   const C_composite_definition_body& rv)
   : C_production(rv), _statementList(0)
{
   if (rv._statementList) {
      _statementList = rv._statementList->duplicate();
   }
}


C_composite_definition_body::C_composite_definition_body(
   C_composite_statement_list *c, SyntaxError * error)
   : C_production(error), _statementList(c)
{
}


C_composite_definition_body* C_composite_definition_body::duplicate() const
{
   return new C_composite_definition_body(*this);
}

void C_composite_definition_body::duplicate(
   std::auto_ptr<RepertoireFactory>& rv) const
{
   rv.reset(new C_composite_definition_body(*this));
}

C_composite_definition_body::~C_composite_definition_body()
{
   delete _statementList;
}


Repertoire* C_composite_definition_body::createRepertoire(
   const std::string& repName, LensContext* c)
{
   Repertoire *composite = new Repertoire(repName);

   // Get current repertoire to set it as parent
   std::string currentRep("CurrentRepertoire");
   const RepertoireDataItem* crdi = 
      dynamic_cast<const RepertoireDataItem*>(
	 c->symTable.getEntry(currentRep));
   if (crdi == 0) {
      std::string mes = 
	 "dynamic cast of DataItem to RepertoireDataItem failed";
      throwError(mes);
   }
   Repertoire* parentRep = crdi->getRepertoire();

   std::auto_ptr<Repertoire> rap(composite);
   parentRep->addSubRepertoire(rap);
   //composite->setParentRepertoire(parentRep);

   // Put new repertoire in symbol table in the current scope
   RepertoireDataItem * rdi = new RepertoireDataItem;
   rdi->setRepertoire(composite);
   std::auto_ptr<DataItem> di_ap(rdi);

   try {
      c->symTable.addEntry(repName, di_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While adding composite definition body " + 
		 repName + " " + e.getError());
   }
   c->symTable.addLocalScope();
   try {
      c->addCurrentRepertoire(composite);
   } catch (SyntaxErrorException& e) {
      throwError("While adding composite definition body " + 
		 currentRep + " " + e.getError());
   }

   // Create a copy so that the original does not get modified.
   C_composite_statement_list* localCopy = 
      new C_composite_statement_list(*_statementList);
   try {
      localCopy->execute(c);
   } catch (SyntaxErrorException& e) {
      e.printError();
      e.resetError();
      setError();
      localCopy->recursivePrint();
      localCopy->printTdError();
      throw;
   }
   c->symTable.removeLocalScope();
   delete localCopy;

   return composite;
}

void C_composite_definition_body::checkChildren() 
{
   if (_statementList) {
      _statementList->checkChildren();
      if (_statementList->isError()) {
         setError();
      }
   }
} 

void C_composite_definition_body::recursivePrint() 
{
   if (_statementList) {
      _statementList->recursivePrint();
   }
   printErrorMessage();
} 

void C_composite_definition_body::setTdError(SyntaxError *tdError)
{
   if (_statementList) {
      _statementList->setTdError(tdError);
   }
}

