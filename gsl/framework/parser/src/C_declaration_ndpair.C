// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_declaration_ndpair.h"
#include "NDPair.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "C_ndpair_clause.h"
#include "NDPairDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

#include <memory>

void C_declaration_ndpair::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   _ndp_clause->execute(c);

   // now transfer to DataItem

   std::auto_ptr<NDPair> ndp;
   _ndp_clause->releaseNDPair(ndp);
   NDPairDataItem *nvdi = new NDPairDataItem;
   nvdi->setNDPair(ndp);
   std::auto_ptr<DataItem> nvdi_ap(nvdi);
   try {
      c->symTable.addEntry(_declarator->getName(), nvdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring ndpair, " + e.getError());
   }
}


C_declaration_ndpair* C_declaration_ndpair::duplicate() const
{
   return new C_declaration_ndpair(*this);
}


C_declaration_ndpair::C_declaration_ndpair(const C_declaration_ndpair& rv)
   : C_declaration(rv), _declarator(0), _ndp_clause(0)
{

   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._ndp_clause) {
      _ndp_clause = rv._ndp_clause->duplicate();
   }
}


C_declaration_ndpair::C_declaration_ndpair(
   C_declarator* d, C_ndpair_clause* nv, SyntaxError* error)
   : C_declaration(error), _declarator(d), _ndp_clause(nv)
{
}


C_declaration_ndpair::~C_declaration_ndpair()
{
   delete _declarator;
   delete _ndp_clause;
}

void C_declaration_ndpair::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_ndp_clause) {
      _ndp_clause->checkChildren();
      if (_ndp_clause->isError()) {
         setError();
      }
   }
} 

void C_declaration_ndpair::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_ndp_clause) {
      _ndp_clause->recursivePrint();
   }
   printErrorMessage();
} 
