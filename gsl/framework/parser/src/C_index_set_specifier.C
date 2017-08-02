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

#include "C_index_set_specifier.h"
#include "C_index_set.h"
#include "C_index_entry.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_index_set_specifier::internalExecute(LensContext *c)
{
   _indexSet->execute(c);

   _indices.clear();
   const std::list<C_index_entry *>& lie = _indexSet->getIndexEntryList();
   std::list<C_index_entry *>::const_iterator iter, end = lie.end();
   for(iter = lie.begin(); iter != end; ++iter) {
      int from = (*iter)->getFrom();
      int to = (*iter)->getTo();
      for (int i = from; i <= to; ++i)
         _indices.push_back(i);
   }
}


const std::vector<int>& C_index_set_specifier::getIndices() const
{
   return _indices;
}


C_index_set_specifier::C_index_set_specifier(const C_index_set_specifier& rv)
   : C_production(rv), _indexSet(0), _indices(rv._indices)
{
   if (rv._indexSet) {
      _indexSet = rv._indexSet->duplicate();
   }
}

C_index_set_specifier::C_index_set_specifier(
   C_index_set *s, SyntaxError * error)
   : C_production(error), _indexSet(s)
{
}

C_index_set_specifier* C_index_set_specifier::duplicate() const
{
   return new C_index_set_specifier(*this);
}


C_index_set_specifier::~C_index_set_specifier()
{
   delete _indexSet;
}

void C_index_set_specifier::checkChildren() 
{
   if (_indexSet) {
      _indexSet->checkChildren();
      if (_indexSet->isError()) {
         setError();
      }
   }
} 

void C_index_set_specifier::recursivePrint() 
{
   if (_indexSet) {
      _indexSet->recursivePrint();
   }
   printErrorMessage();
} 
