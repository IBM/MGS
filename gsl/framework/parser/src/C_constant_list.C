// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_constant_list.h"
#include "C_constant.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_constant_list::internalExecute(GslContext *c)
{
   std::list<C_constant>::iterator iter, end = _list->end();
   for(iter = _list->begin(); iter != end; ++iter) {
      iter->execute(c);
   }
}


std::list<C_constant>* C_constant_list::releaseList()
{
   std::list<C_constant> *retval = _list;
   _list = 0;
   return retval;
}


C_constant_list::C_constant_list(const C_constant_list& rv)
   : C_production(rv), _list(0)
{
   if (rv._list) {
      _list = new std::list<C_constant>(*rv._list);
   }
}


C_constant_list::C_constant_list(C_constant *c, SyntaxError * error)
   : C_production(error), _list(0)
{
   _list = new std::list<C_constant>;
   _list->push_back(*c);
   delete c;
}


C_constant_list::C_constant_list(
   C_constant_list *cl, C_constant *c, SyntaxError * error)
   : C_production(error), _list(0)
{
   if (cl) {
      if (cl->isError()) {
	 delete _error;
	 _error = cl->_error->duplicate();
      }
      _list = cl->releaseList();
      if (c) _list->push_back(*c);
   }
   delete cl;
   delete c;
}


C_constant_list* C_constant_list::duplicate() const
{
   return new C_constant_list(*this);
}


std::list<C_constant> * C_constant_list::getList() const
{
   return _list;
}


C_constant_list::~C_constant_list()
{
   delete _list;
}

void C_constant_list::checkChildren() 
{
   if (_list) {
      std::list<C_constant>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 i->checkChildren();
	 if (i->isError()) {
	    setError();
	 }
      }
   }
} 

void C_constant_list::recursivePrint() 
{
   if (_list) {
      std::list<C_constant>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 i->recursivePrint();
      }
   }
   printErrorMessage();
} 
