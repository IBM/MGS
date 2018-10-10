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

#include "C_ndpair_clause_list_body.h"
#include "NDPairList.h"
#include "C_ndpair_clause.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_ndpair_clause_list_body::internalExecute(LensContext *c)
{
   delete _ndpairList;
   _ndpairList = new NDPairList;
   std::unique_ptr<NDPair> ndp;
   std::list<C_ndpair_clause*>::iterator i, end = _list->end();
   for(i = _list->begin(); i != end; ++i) {
      (*i)->execute(c);
      (*i)->releaseNDPair(ndp);
      _ndpairList->push_back(ndp.release());
   }
}

C_ndpair_clause_list_body::C_ndpair_clause_list_body(
   const C_ndpair_clause_list_body& rv)
   : C_production(rv), _ndpairList(0), _list(0)
{
   _list = new std::list<C_ndpair_clause*>;
   if (rv._list) {
      std::list<C_ndpair_clause *>::iterator i, end = rv._list->end();
      for (i = rv._list->begin(); i != end; ++i) {
         _list->push_back((*i)->duplicate());
      }
   }

   if (rv._ndpairList) {
      _ndpairList = new NDPairList(*rv._ndpairList);
   }
}

C_ndpair_clause_list_body::C_ndpair_clause_list_body(
   C_ndpair_clause *c, SyntaxError * error)
   : C_production(error), _ndpairList(0), _list(0)
{
   _list = new std::list<C_ndpair_clause*>;
   _list->push_back(c);
}

C_ndpair_clause_list_body::C_ndpair_clause_list_body(
   C_ndpair_clause *c, C_ndpair_clause_list_body *clb, SyntaxError * error)
   : C_production(error), _ndpairList(0), _list(0)
{
   if (clb) {
      if (clb->isError()) {
	 delete _error;
	 _error = clb->_error->duplicate();
      } 
      _list = clb->releaseList();
      if (c) _list->push_back(c);
   } 
   delete clb;
}

std::list<C_ndpair_clause*>* C_ndpair_clause_list_body::releaseList()
{
   std::list<C_ndpair_clause*> *retval = _list;
   _list = 0;
   return retval;
}

C_ndpair_clause_list_body* C_ndpair_clause_list_body::duplicate() const
{
   return new C_ndpair_clause_list_body(*this);
}

C_ndpair_clause_list_body::~C_ndpair_clause_list_body()
{
   if (_list) {
      std::list<C_ndpair_clause*>::iterator i, end = _list->end();
      for(i = _list->begin(); i != end; ++i)
         delete *i;
   }
   delete _list;
   delete _ndpairList;

}

void C_ndpair_clause_list_body::checkChildren() 
{
   if (_list) {
      std::list<C_ndpair_clause*>::iterator i, begin, end;
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

void C_ndpair_clause_list_body::recursivePrint() 
{
   if (_list) {
      std::list<C_ndpair_clause*>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 
