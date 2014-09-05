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

#include "C_declaration_node_type_set.h"
#include "C_node_type_set_specifier_clause.h"
#include "LensContext.h"
#include "Grid.h"
#include "C_declarator.h"
#include "NodeTypeSetDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_node_type_set::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   _nodeTypeSetSpecifierClause->execute(c);

   NodeTypeSetDataItem *ntsdi = new NodeTypeSetDataItem;
   ntsdi->setNodeTypeSet(_nodeTypeSetSpecifierClause);

   std::auto_ptr<DataItem> ntsdi_ap(ntsdi);
   try {
      c->symTable.addEntry(_declarator->getName(), ntsdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring node type set, " + e.getError());
   }
}


C_declaration_node_type_set* C_declaration_node_type_set::duplicate() const
{
   return new C_declaration_node_type_set(*this);
}


C_declaration_node_type_set::C_declaration_node_type_set(
   C_declarator *d, C_node_type_set_specifier_clause *n, SyntaxError * error)
   : C_declaration(error), _declarator(d), _nodeTypeSetSpecifierClause(n)
{
}


C_declaration_node_type_set::C_declaration_node_type_set(
   const C_declaration_node_type_set& rv)
   : C_declaration(rv), _declarator(0), _nodeTypeSetSpecifierClause(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._nodeTypeSetSpecifierClause) {
      _nodeTypeSetSpecifierClause = rv._nodeTypeSetSpecifierClause->duplicate();
   }
}


C_declaration_node_type_set::~C_declaration_node_type_set()
{
   delete _declarator;
   delete _nodeTypeSetSpecifierClause;
}

void C_declaration_node_type_set::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_nodeTypeSetSpecifierClause) {
      _nodeTypeSetSpecifierClause->checkChildren();
      if (_nodeTypeSetSpecifierClause->isError()) {
         setError();
      }
   }
} 

void C_declaration_node_type_set::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_nodeTypeSetSpecifierClause) {
      _nodeTypeSetSpecifierClause->recursivePrint();
   }
   printErrorMessage();
} 
