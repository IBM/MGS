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

#include "C_argument_list.h"
#include "C_argument.h"
#include "DataItem.h"
#include <assert.h>
#include "SyntaxError.h"
#include "C_production.h"

void C_argument_list::internalExecute(LensContext *c)
{
   _dataitem_list.clear();
   std::list<C_argument*>::iterator iter, end = _list->end();
   for (iter = _list->begin(); iter != end; ++iter) {
      (*iter)->execute(c);
      DataItem* di = (*iter)->getArgumentDataItem();
      if (di) {
	 _dataitem_list.push_back(di);
      }
   }
}


C_argument_list::C_argument_list(const C_argument_list& rv)
   : C_production(rv), _dataitem_list(rv._dataitem_list)
{
   _list = new std::list<C_argument*>;
   if (rv._list) {
      std::list<C_argument*>::const_iterator iter, begin = rv._list->begin(),
	 end = rv._list->end();
      for (iter=begin; iter != end; ++iter) {
	 _list->push_back((*iter)->duplicate());
      }
   }
}


C_argument_list::C_argument_list(C_argument *a, SyntaxError * error)
   : C_production(error), _list(0)
{
   _list = new std::list<C_argument*>;
   _list->push_back(a);
}


C_argument_list::C_argument_list(C_argument_list *al, C_argument *a, 
				 SyntaxError * error)
   : C_production(error), _list(0)
{
   if (al) {
      if (al->isError()) {
	 delete _error;
	 _error = al->_error->duplicate();
      }
      _list = al->releaseList();
      if (a) _list->push_back(a);
   }
   delete al;
}


std::list<C_argument*> *C_argument_list::releaseList()
{
   std::list<C_argument*>* retval = _list;
   _list = 0;
   return retval;
}


C_argument_list* C_argument_list::duplicate() const
{
   return new C_argument_list(*this);
}


C_argument_list::~C_argument_list()
{
   if (_list) { // Careful {} creates a local scope for iter and end
      std::list<C_argument*>::iterator iter, end = _list->end();
      for (iter = _list->begin(); iter != end; ++iter ) {
         delete *iter;
      }
   }
   // do not delete _dataitem_list, they are deleted by the arguments.
   delete _list;
}

void C_argument_list::checkChildren() 
{
   if (_list) {
      std::list<C_argument*>::iterator i, begin, end;
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

void C_argument_list::recursivePrint() 
{
   if (_list) {
      std::list<C_argument*>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 
