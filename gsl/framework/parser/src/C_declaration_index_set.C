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

#include "C_declaration_index_set.h"
#include "LensContext.h"
#include "C_index_set.h"
#include "C_declarator.h"
#include "C_index_entry.h"
#include "IndexSet.h"
#include "IndexSetDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_index_set::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   _indexSet->execute(c);

   std::vector<int> beginVec;
   std::vector<int> endVec;

   const std::list<C_index_entry *>& lie = _indexSet->getIndexEntryList();
   std::list<C_index_entry *>::const_iterator iter, end = lie.end();
   for(iter = lie.begin(); iter != end; ++iter) {
      beginVec.push_back((*iter)->getFrom());
      endVec.push_back((*iter)->getTo());
   }

   IndexSet* indexSet = new IndexSet(beginVec, endVec);
   IndexSetDataItem* isdi = new IndexSetDataItem();
   isdi->setIndexSet(indexSet);

   std::auto_ptr<DataItem> di_ap(isdi);
   try {
      c->symTable.addEntry(_declarator->getName(), di_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring index set, " + e.getError());
   }
}


C_declaration_index_set* C_declaration_index_set::duplicate() const
{
   return new C_declaration_index_set(*this);
}


C_index_set* C_declaration_index_set::getIndexSet()
{
   return _indexSet;
}


C_declaration_index_set::C_declaration_index_set(
   const C_declaration_index_set& rv)
   : C_declaration(rv), _declarator(0), _indexSet(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._indexSet) {
      _indexSet = rv._indexSet->duplicate();
   }
}


C_declaration_index_set::C_declaration_index_set(
   C_declarator *d, C_index_set *n, SyntaxError * error)
   :  C_declaration(error), _declarator(d), _indexSet(n)
{
}


C_declaration_index_set::~C_declaration_index_set()
{
   delete _declarator;
   delete _indexSet;
}

void C_declaration_index_set::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_indexSet) {
      _indexSet->checkChildren();
      if (_indexSet->isError()) {
         setError();
      }
   }
} 

void C_declaration_index_set::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_indexSet) {
      _indexSet->recursivePrint();
   }
   printErrorMessage();
} 
