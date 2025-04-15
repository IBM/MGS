// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_composite_statement_list.h"
#include "C_composite_statement.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "C_production.h"
//#include "LensLexer.h"

void C_composite_statement_list::internalExecute(LensContext *c)
{
   std::list<C_composite_statement*>::iterator i, end = _list->end();
   for(i = _list->begin(); i != end; ++i) {
      (*i)->execute(c);
   }
}


C_composite_statement_list::C_composite_statement_list(
   const C_composite_statement_list& rv)
   : C_production(rv), _list(0), _tdError(0)
{
   _list = new std::list<C_composite_statement*>;
   std::list<C_composite_statement*>::iterator i, end;
   if (rv._list) {
      end = rv._list->end();
      for(i = rv._list->begin(); i != end; ++i)
      _list->push_back((*i)->duplicate());
   }
   if (rv._tdError) {
      _tdError = rv._tdError->duplicate();
   }
}


C_composite_statement_list::C_composite_statement_list(
   C_composite_statement *c, SyntaxError * error)
   : C_production(error), _list(0), _tdError(0)
{
   _list = new std::list<C_composite_statement*>;
   _list->push_back(c);
   _tdError = 0;
}


C_composite_statement_list::C_composite_statement_list(
   C_composite_statement_list *cl, C_composite_statement *cs, 
   SyntaxError * error)
   : C_production(error), _list(0), _tdError(0)
{
   if (cl) {
      if (cl->isError()) {
	 delete _error;
	 _error = cl->_error->duplicate();
      }
      _list = cl->releaseList();
      if (cs) _list->push_back(cs);
   }
   delete cl;
   _tdError = 0;
}


std::list<C_composite_statement*>* C_composite_statement_list::releaseList()
{
   std::list<C_composite_statement*>* retval = _list;
   _list = 0;
   return retval;
}


C_composite_statement_list* C_composite_statement_list::duplicate() const
{
   return new C_composite_statement_list(*this);
}


C_composite_statement_list::~C_composite_statement_list()
{
   if (_list) {
      std::list<C_composite_statement*>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i)
         delete *i;
      delete _list;
   }
   delete _tdError;
}

void C_composite_statement_list::checkChildren() 
{
   if (_list) {
      std::list<C_composite_statement*>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 (*i)->checkChildren();
	 if ((*i)->isError()) {
	    setError();
	 }
      }
   }
} 

void C_composite_statement_list::recursivePrint() 
{
   if (_list) {
      std::list<C_composite_statement*>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 

void C_composite_statement_list::printTdError() 
{
   _tdError->setOriginal();
   _tdError->setError(true);   
   if (_tdError) {
      _tdError->printMessage();
   }
}
