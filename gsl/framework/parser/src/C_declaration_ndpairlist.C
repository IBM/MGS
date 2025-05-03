// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_ndpairlist.h"
#include "GslContext.h"
#include "C_declarator.h"
#include "NDPairDataItem.h"
#include "NDPairListDataItem.h"
#include "NDPairList.h"
#include "C_ndpair_clause_list.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_ndpairlist::internalExecute(GslContext *c)
{
   _declarator->execute(c);
   _ndp_clause_list->execute(c);

   std::unique_ptr<NDPairList> ndp;
   _ndp_clause_list->releaseList(ndp);   
   NDPairListDataItem *nvl = new NDPairListDataItem;
   nvl->setNDPairList(ndp);
   std::unique_ptr<DataItem> nvl_ap(nvl);

   try {
      c->symTable.addEntry(_declarator->getName(), nvl_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring ndpairlist, " + e.getError());
   }
}


C_declaration_ndpairlist* C_declaration_ndpairlist::duplicate() const
{
   return new C_declaration_ndpairlist(*this);
}


C_declaration_ndpairlist::C_declaration_ndpairlist(
   const C_declaration_ndpairlist& rv)
   : C_declaration(rv), _declarator(0), _ndp_clause_list(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._ndp_clause_list) {
      _ndp_clause_list = rv._ndp_clause_list->duplicate();
   }
}


C_declaration_ndpairlist::C_declaration_ndpairlist( 
   C_declarator *d, C_ndpair_clause_list *nv, SyntaxError * error)
   : C_declaration(error), _declarator(d), _ndp_clause_list(nv)
{
}


C_declaration_ndpairlist::~C_declaration_ndpairlist()
{
   delete _declarator;
   delete _ndp_clause_list;
}

void C_declaration_ndpairlist::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_ndp_clause_list) {
      _ndp_clause_list->checkChildren();
      if (_ndp_clause_list->isError()) {
         setError();
      }
   }
} 

void C_declaration_ndpairlist::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_ndp_clause_list) {
      _ndp_clause_list->recursivePrint();
   }
   printErrorMessage();
} 
