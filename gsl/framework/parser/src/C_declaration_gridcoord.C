// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_declaration_gridcoord.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "C_gridset.h"
#include "GridSet.h"
#include "NodeSetDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"


void C_declaration_gridcoord::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   _gridset->execute(c);
   //GridSetDataItem *gsdi = new GridSetDataItem;
   //gsdi->setGridSet(_gridset->getGridSet());
   NodeSetDataItem *gsdi = new NodeSetDataItem;
   gsdi->setNodeSet(_gridset->getGridSet());
   std::auto_ptr<DataItem> di_ap(static_cast<DataItem*>(gsdi));
   try {
      c->symTable.addEntry(_declarator->getName(), di_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring gridcoord, " + e.getError());
   }
}


C_declaration_gridcoord * C_declaration_gridcoord::duplicate() const
{
   return new C_declaration_gridcoord(*this);
}


C_declaration_gridcoord::C_declaration_gridcoord(
   const C_declaration_gridcoord& rv)
   : C_declaration(rv), _declarator(0), _gridset(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._gridset) {
      _gridset = rv._gridset->duplicate();
   }
}


C_declaration_gridcoord::C_declaration_gridcoord( 
   C_declarator *d, C_gridset *s, SyntaxError * error)
   : C_declaration(error), _declarator(d), _gridset(s)
{
}


C_declaration_gridcoord::~C_declaration_gridcoord()
{
   delete _declarator;
   delete _gridset;
}

void C_declaration_gridcoord::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_gridset) {
      _gridset->checkChildren();
      if (_gridset->isError()) {
         setError();
      }
   }
} 

void C_declaration_gridcoord::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_gridset) {
      _gridset->recursivePrint();
   }
   printErrorMessage();
} 
