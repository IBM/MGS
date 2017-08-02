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

#include "C_declaration_edgeset.h"
#include "EdgeSet.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "C_edgeset.h"
#include "EdgeSetDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_edgeset::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   _edgeset->execute(c);

   EdgeSetDataItem *esdi = new EdgeSetDataItem;
   std::auto_ptr<DataItem> esdi_ap(esdi);

   EdgeSet *ns = _edgeset->getEdgeSet();
   esdi->setEdgeSet(ns);

   try {
      c->symTable.addEntry(_declarator->getName(), esdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring edgeset, " + e.getError());
   }
}


C_declaration_edgeset* C_declaration_edgeset::duplicate() const
{
   return new C_declaration_edgeset(*this);
}


C_declaration_edgeset::C_declaration_edgeset(C_declarator *d, C_edgeset *n, 
					     SyntaxError * error)
   : C_declaration(error), _declarator(d), _edgeset(n)
{
}


C_declaration_edgeset::C_declaration_edgeset(const C_declaration_edgeset& rv)
   : C_declaration(rv), _declarator(0), _edgeset(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._edgeset) {
      _edgeset = rv._edgeset->duplicate();
   }
}


C_declaration_edgeset::~C_declaration_edgeset()
{
   delete _declarator;
   delete _edgeset;
}

void C_declaration_edgeset::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_edgeset) {
      _edgeset->checkChildren();
      if (_edgeset->isError()) {
         setError();
      }
   }
} 

void C_declaration_edgeset::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_edgeset) {
      _edgeset->recursivePrint();
   }
   printErrorMessage();
} 
