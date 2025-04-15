// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_repname.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "C_repname.h"
#include "RepertoireDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include <memory>

void C_declaration_repname::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   _repname->execute(c);

   Repertoire *rep = _repname->getRepertoire();
   if(rep) {
      RepertoireDataItem *rpdi = new RepertoireDataItem;
      std::unique_ptr<DataItem> rp_ap(rpdi);
      rpdi->setRepertoire(rep);
      try {
	 c->symTable.addEntry(_declarator->getName(), rp_ap);
      } catch (SyntaxErrorException& e) {
	 throwError("While declaring repname, " + e.getError());
      }
   }
   else {
      std::string mes = "unable to declare repname";
      throwError(mes);
   }
}


C_declaration_repname* C_declaration_repname::duplicate() const
{
   return new C_declaration_repname(*this);
}


C_declaration_repname::C_declaration_repname(const C_declaration_repname& rv)
   : C_declaration(rv), _declarator(0), _repname(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._repname) {
      _repname = rv._repname->duplicate();
   }
}


C_declaration_repname::~C_declaration_repname()
{
   delete _declarator;
   delete _repname;
}


C_declaration_repname::C_declaration_repname(
   C_declarator *d, C_repname *rp, SyntaxError * error)
  : C_declaration(error), _declarator(d), _repname(rp)
{
}

void C_declaration_repname::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_repname) {
      _repname->checkChildren();
      if (_repname->isError()) {
         setError();
      }
   }
} 

void C_declaration_repname::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_repname) {
      _repname->recursivePrint();
   }
   printErrorMessage();
} 
