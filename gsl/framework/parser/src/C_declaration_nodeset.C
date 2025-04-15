// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_nodeset.h"
#include "NodeSet.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "C_nodeset.h"
#include "NodeSetDataItem.h"
#include "Grid.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_nodeset::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   _nodeset->execute(c);

   NodeSetDataItem *nsdi = new NodeSetDataItem;
   std::unique_ptr<DataItem> nsdi_ap(nsdi);

   NodeSet *ns = _nodeset->getNodeSet();;
   nsdi->setNodeSet(ns);

   try {
      c->symTable.addEntry(_declarator->getName(), nsdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring nodeset, " + e.getError());
   }
}


C_declaration_nodeset* C_declaration_nodeset::duplicate() const
{
   return new C_declaration_nodeset(*this);
}


C_declaration_nodeset::C_declaration_nodeset(C_declarator *d, C_nodeset *n, 
					     SyntaxError * error)
   : C_declaration(error), _declarator(d), _nodeset(n)
{
}


C_declaration_nodeset::C_declaration_nodeset(const C_declaration_nodeset& rv)
   : C_declaration(rv), _declarator(0), _nodeset(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._nodeset) {
      _nodeset = rv._nodeset->duplicate();
   }
}


C_declaration_nodeset::~C_declaration_nodeset()
{
   delete _declarator;
   delete _nodeset;
}

void C_declaration_nodeset::checkChildren() 
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

void C_declaration_nodeset::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_nodeset) {
      _nodeset->recursivePrint();
   }
   printErrorMessage();
} 
