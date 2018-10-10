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

#include "C_declaration_rel_nodeset.h"
#include "C_declarator.h"
#include "C_relative_nodeset.h"
#include "LensContext.h"
#include "RelativeNodeSetDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_rel_nodeset::internalExecute(LensContext *c)
{
   _declaration->execute(c);
   _relativeNodeSet->execute(c);

   RelativeNodeSetDataItem *rnsdi = new RelativeNodeSetDataItem;
   rnsdi->setRelativeNodeSet(_relativeNodeSet);

   std::unique_ptr<DataItem> rnsdi_ap(rnsdi);
   try {
      c->symTable.addEntry(_declaration->getName(), rnsdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring rel nodeset, " + e.getError());
   }
}


C_declaration_rel_nodeset* C_declaration_rel_nodeset::duplicate() const
{
   return new C_declaration_rel_nodeset(*this);
}


C_declaration_rel_nodeset::C_declaration_rel_nodeset(
   C_declarator *d, C_relative_nodeset *n, SyntaxError * error)
   : C_declaration(error), _declaration(d), _relativeNodeSet(n)
{
}


C_declaration_rel_nodeset::C_declaration_rel_nodeset(
   const C_declaration_rel_nodeset& rv)
   : C_declaration(rv), _declaration(0), _relativeNodeSet(0)
{
   if (rv._declaration) {
      _declaration = rv._declaration->duplicate();
   }
   if (rv._relativeNodeSet) {
      _relativeNodeSet = rv._relativeNodeSet->duplicate();
   }
}


C_declaration_rel_nodeset::~C_declaration_rel_nodeset()
{
   delete _declaration;
   delete _relativeNodeSet;
}

void C_declaration_rel_nodeset::checkChildren() 
{
   if (_declaration) {
      _declaration->checkChildren();
      if (_declaration->isError()) {
         setError();
      }
   }
   if (_relativeNodeSet) {
      _relativeNodeSet->checkChildren();
      if (_relativeNodeSet->isError()) {
         setError();
      }
   }
} 

void C_declaration_rel_nodeset::recursivePrint() 
{
   if (_declaration) {
      _declaration->recursivePrint();
   }
   if (_relativeNodeSet) {
      _relativeNodeSet->recursivePrint();
   }
   printErrorMessage();
} 
