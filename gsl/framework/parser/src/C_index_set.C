// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_index_set.h"
#include "C_index_entry.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_index_set::internalExecute(GslContext *c)
{
   std::list<C_index_entry*>::iterator iter, end = _listIndexEntry->end();
   for(iter = _listIndexEntry->begin(); iter != end; ++iter) {
      (*iter)->execute(c);
   }
}

C_index_set::C_index_set(const C_index_set& rv)
   : C_production(rv), _listIndexEntry(0)
{
   _listIndexEntry = new std::list<C_index_entry*>();
   if(rv._listIndexEntry) {
      std::list<C_index_entry*>::iterator iter, 
	 end = rv._listIndexEntry->end();
      for (iter = rv._listIndexEntry->begin(); iter != end; ++iter) {
	 _listIndexEntry->push_back((*iter)->duplicate());
      }
   }
}

C_index_set::C_index_set(C_index_entry* entry, SyntaxError* error)
   : C_production(error), _listIndexEntry(0)
{
   _listIndexEntry = new std::list<C_index_entry *>;
   _listIndexEntry->push_back(entry);
}


C_index_set::C_index_set(C_index_set* st, C_index_entry* entry, 
			 SyntaxError* error)
   : C_production(error), _listIndexEntry(0)
{
   if (st) {
      if (st->isError()) {
	 delete _error;
	 _error = st->_error->duplicate();
      }
      _listIndexEntry = st->releaseList();
      if (entry != 0) _listIndexEntry->push_back(entry);
   }
   delete st;
}

C_index_set::C_index_set(SyntaxError * error)
   : C_production(error), _listIndexEntry(0)
{
   _listIndexEntry = new std::list<C_index_entry *>;
}


C_index_set* C_index_set::duplicate() const
{
   return new C_index_set(*this);
}


std::list<C_index_entry*>* C_index_set::releaseList()
{
   std::list<C_index_entry*> *retval = _listIndexEntry;
   _listIndexEntry = 0;
   return retval;
}


const std::list<C_index_entry *>& C_index_set::getIndexEntryList() const
{
   return *_listIndexEntry;
}


C_index_set::~C_index_set()
{
   if (_listIndexEntry) {
      std::list<C_index_entry*>::iterator i, begin, end;
      begin =_listIndexEntry->begin();
      end =_listIndexEntry->end();
      for(i=begin;i!=end;++i) {
	 delete *i;
      }
   }
   delete _listIndexEntry;
}

void C_index_set::checkChildren() 
{
   if (_listIndexEntry) {
      std::list<C_index_entry*>::iterator i, begin, end;
      begin =_listIndexEntry->begin();
      end =_listIndexEntry->end();
      for(i=begin;i!=end;++i) {
	 (*i)->checkChildren();
	 if ((*i)->isError()) {
	    setError();
	 }
      }
   }
} 

void C_index_set::recursivePrint() 
{
   if (_listIndexEntry) {
      std::list<C_index_entry*>::iterator i, begin, end;
      begin =_listIndexEntry->begin();
      end =_listIndexEntry->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 
