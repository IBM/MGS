// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_gridnodeset.h"
#include "C_index_set.h"
#include "C_index_entry.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_gridnodeset::internalExecute(LensContext *c)
{
   _begin.clear();
   _end.clear();
   _indexSet->execute(c);
   const std::list<C_index_entry *>& entries = _indexSet->getIndexEntryList();
   std::list<C_index_entry *>::const_reverse_iterator i, 
      rbegin = entries.rbegin(), rend = entries.rend();
   for (i = rbegin; i != rend; ++i) {
      _begin.push_back((*i)->getFrom());
      _increment.push_back((*i)->getIncrement());
      _end.push_back((*i)->getTo());
   }

}


C_gridnodeset::C_gridnodeset(const C_gridnodeset& rv)
   : C_production(rv), _indexSet(0), _begin(rv._begin), _increment(rv._increment), _end(rv._end)
{
   if (rv._indexSet) {
      _indexSet = rv._indexSet->duplicate();
   }
}


C_gridnodeset::C_gridnodeset(C_index_set *s, SyntaxError * error)
   : C_production(error), _indexSet(s)
{
}


C_gridnodeset* C_gridnodeset::duplicate() const
{
   return new C_gridnodeset(*this);
}


C_gridnodeset::~C_gridnodeset()
{
   delete _indexSet;
}

void C_gridnodeset::checkChildren() 
{
   if (_indexSet) {
      _indexSet->checkChildren();
      if (_indexSet->isError()) {
         setError();
      }
   }
} 

void C_gridnodeset::recursivePrint() 
{
   if (_indexSet) {
      _indexSet->recursivePrint();
   }
   printErrorMessage();
} 
