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

#include "C_ndpair_clause_list.h"
#include "C_ndpair_clause_list_body.h"
#include "NDPairList.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_ndpair_clause_list::internalExecute(LensContext *c)
{
   delete _list;
   _list = new NDPairList;
   if(_bodyList) {
      std::list<C_ndpair_clause_list_body*>::iterator i, 
	 end = _bodyList->end();
      for(i = _bodyList->begin(); i != end; ++i) {
         (*i)->execute(c);
         _list->splice(_list->end(), *(*i)->getNDPairList());
      }
   }
}


C_ndpair_clause_list::C_ndpair_clause_list(const C_ndpair_clause_list& rv)
   : C_production(rv), _bodyList(0), _list(0)
{
   if (rv._list) {
      _list = new NDPairList(*rv._list);
   }
   if (rv._bodyList) {
      _bodyList = new std::list<C_ndpair_clause_list_body*>;
      std::list<C_ndpair_clause_list_body*>::iterator i,
	 end = rv._bodyList->end();
      for(i = rv._bodyList->begin(); i != end; ++i)
         _bodyList->push_back((*i)->duplicate());
   }

}


C_ndpair_clause_list::C_ndpair_clause_list(C_ndpair_clause_list_body *nvclb, 
					   SyntaxError * error)
   : C_production(error), _bodyList(0), _list(0)
{
   _bodyList = new std::list<C_ndpair_clause_list_body*>();
   _bodyList->push_back(nvclb);
}


C_ndpair_clause_list::C_ndpair_clause_list(
   C_ndpair_clause_list *nvcl, C_ndpair_clause_list_body *nvclb, 
   SyntaxError * error)
   : C_production(error), _bodyList(0), _list(0)
{
   if (nvcl) {
      if (nvcl->isError()) {
	 delete _error;
	 _error = nvcl->_error->duplicate();
      }
      _bodyList = nvcl->releaseList();
      if (nvclb) _bodyList->push_back(nvclb);
   }
   delete nvcl;
}


C_ndpair_clause_list::C_ndpair_clause_list(SyntaxError * error)
   : C_production(error), _bodyList(0), _list(0)
{
}


std::list<C_ndpair_clause_list_body*>* C_ndpair_clause_list::releaseList()
{
   std::list<C_ndpair_clause_list_body*> *retval = _bodyList;
   _bodyList =0;
   return retval;
}


C_ndpair_clause_list* C_ndpair_clause_list::duplicate() const
{
   return new C_ndpair_clause_list(*this);
}

void C_ndpair_clause_list::releaseList(std::auto_ptr<NDPairList>& ndp)
{
   ndp.reset(_list);
   _list = 0;
}

C_ndpair_clause_list::~C_ndpair_clause_list()
{
   if (_bodyList) {
      std::list<C_ndpair_clause_list_body*>::iterator i,
	 end = _bodyList->end();
      for(i = _bodyList->begin(); i != end; ++i)
         delete *i;
      delete _bodyList;
   }
   delete _list;
}

void C_ndpair_clause_list::checkChildren() 
{
   if (_bodyList) {
      std::list<C_ndpair_clause_list_body*>::iterator i, begin, end;
      begin =_bodyList->begin();
      end =_bodyList->end();
      for(i=begin;i!=end;++i) {
	 (*i)->checkChildren();
	 if ((*i)->isError()) {
	    setError();
	 }
      }
   }
} 

void C_ndpair_clause_list::recursivePrint() 
{
   if (_bodyList) {
      std::list<C_ndpair_clause_list_body*>::iterator i, begin, end;
      begin =_bodyList->begin();
      end =_bodyList->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 
