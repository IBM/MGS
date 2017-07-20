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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "C_parameter_type_list.h"
#include "C_parameter_type.h"
#include "SyntaxError.h"
#include "C_production.h"
#include <iostream>


void C_parameter_type_list::internalExecute(LensContext *c)
{

   std::list<C_parameter_type>::iterator i,end=_list->end();
   for(i=_list->begin();i!=end;++i)
      i->execute(c);
}


C_parameter_type_list::C_parameter_type_list(const C_parameter_type_list& rv)
   : C_production(rv), _list(0)
{
   if (rv._list) {
      _list = new std::list<C_parameter_type>(*rv._list);
   }
}

C_parameter_type_list::C_parameter_type_list(SyntaxError * error)
   : C_production(error), _list(0)
{
}

C_parameter_type_list::C_parameter_type_list(
   C_parameter_type_list *p, C_parameter_type *t, SyntaxError * error)
   : C_production(error), _list(0)
{
   if (p) {
      if (p->isError()) {
	 delete _error;
	 _error = p->_error->duplicate();
      }
      if (p->_list) {
	 _list = p->releaseList();
	 if (!_list->rbegin()->isSpecified()) {
	    std::cerr << "Illegal parameter list: no parameters may follow an elipsis." << std::endl;
	 }
	 if (t) _list->push_back(*t);
      } 
   }
   delete p;   
   delete t;
}


C_parameter_type_list::C_parameter_type_list(
   C_parameter_type *p, SyntaxError * error)
   : C_production(error), _list(0)
{
   _list = new std::list<C_parameter_type>;
   _list->push_back(*p);
   if (p->isError()) {
      setError();
   }
   delete p;
}


std::list<C_parameter_type>* C_parameter_type_list::releaseList()
{
   std::list<C_parameter_type>* retval = _list;
   _list = 0;
   return retval;
}


C_parameter_type_list* C_parameter_type_list::duplicate() const
{
   return new C_parameter_type_list(*this);
}


C_parameter_type_list::~C_parameter_type_list()
{
   delete _list;
}

void C_parameter_type_list::checkChildren() 
{
   if (_list) {
      std::list<C_parameter_type>::iterator i, begin, end;
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

void C_parameter_type_list::recursivePrint() 
{
   if (_list) {
      std::list<C_parameter_type>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 i->recursivePrint();
      }
   }
   printErrorMessage();
} 
