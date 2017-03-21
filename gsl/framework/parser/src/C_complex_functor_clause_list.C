// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_complex_functor_clause_list.h"
#include "C_complex_functor_clause.h"
#include "SyntaxError.h"
#include "C_production.h"


void C_complex_functor_clause_list::internalExecute(LensContext *c)
{
   std::list<C_complex_functor_clause*>::iterator i,end=_list->end();
   for(i = _list->begin(); i != end; ++i) {
      (*i)->execute(c);
   }
}

C_complex_functor_clause_list::C_complex_functor_clause_list(
   const C_complex_functor_clause_list& rv)
   : C_production(rv), _list(0)
{
   _list = new std::list<C_complex_functor_clause*>;
   if (rv._list) {
      std::list<C_complex_functor_clause*>::iterator i, end = rv._list->end();
      for (i = rv._list->begin(); i != end; ++i) {
	 _list->push_back((*i)->duplicate());
      }
   }
}


C_complex_functor_clause_list::C_complex_functor_clause_list(
   C_complex_functor_clause_list *p, C_complex_functor_clause *t, 
   SyntaxError * error)
   : C_production(error), _list(0)
{
   if (p) {
      if (p->isError())	{
	 delete _error;
	 _error = p->_error->duplicate();
      }
      _list = p->releaseList();   
      if (t != 0) _list->push_back(t);
   }
   delete p;
}


C_complex_functor_clause_list::C_complex_functor_clause_list(
   C_complex_functor_clause *p, SyntaxError * error)
   : C_production(error), _list(0)
{
   _list = new std::list<C_complex_functor_clause*>;
   _list->push_back(p);
}


std::list<C_complex_functor_clause*>* 
C_complex_functor_clause_list::releaseList()
{
   std::list<C_complex_functor_clause*>* retval = _list;
   _list = 0;
   return retval;
}


C_complex_functor_clause_list* C_complex_functor_clause_list::duplicate() const
{
   return new C_complex_functor_clause_list(*this);
}


C_complex_functor_clause_list::~C_complex_functor_clause_list()
{
   if (_list) {
      std::list<C_complex_functor_clause*>::iterator i, end = _list->end();
      for (i = _list->begin(); i != end; i++) {
         delete *i;
      }
      delete _list;
   }
}

void C_complex_functor_clause_list::checkChildren() 
{
   if (_list) {
      std::list<C_complex_functor_clause*>::iterator i, begin, end;
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

void C_complex_functor_clause_list::recursivePrint() 
{
   if (_list) {
      std::list<C_complex_functor_clause*>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 
