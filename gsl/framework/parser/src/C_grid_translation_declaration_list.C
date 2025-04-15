// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_grid_translation_declaration_list.h"
#include "C_grid_translation_declaration.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "C_production_grid.h"


void C_grid_translation_declaration_list::internalExecute(LensContext *c, Grid* g)
{
   std::list<C_grid_translation_declaration*>::iterator i, end =_list->end();
   for(i = _list->begin(); i != end; ++i) {
      (*i)->execute(c,g);
   }
}


C_grid_translation_declaration_list::C_grid_translation_declaration_list(
   const C_grid_translation_declaration_list& rv)
   : C_production_grid(rv), _list(0)
{
   _list = new std::list<C_grid_translation_declaration*>;
   std::list<C_grid_translation_declaration*>::iterator i, end;
   if (rv._list) {
      end = rv._list->end();
      for(i = rv._list->begin(); i != end; ++i)
	 _list->push_back((*i)->duplicate());
   }
}


C_grid_translation_declaration_list::C_grid_translation_declaration_list(
   C_grid_translation_declaration *g, SyntaxError * error)
   : C_production_grid(error), _list(0)
{
   _list = new std::list<C_grid_translation_declaration*>;
   _list->push_back(g);
}


C_grid_translation_declaration_list::C_grid_translation_declaration_list(
   C_grid_translation_declaration_list *gl, C_grid_translation_declaration *g, 
   SyntaxError * error)
   : C_production_grid(error), _list(0)
{
   if (gl) {
      if (gl->isError()) {
	 delete _error;
	 _error = gl->_error->duplicate();
      }
      _list = gl->releaseList();
      if (g != 0) _list->push_back(g);
   }
   delete gl;
}


std::list<C_grid_translation_declaration*>* 
C_grid_translation_declaration_list::releaseList()
{
   std::list<C_grid_translation_declaration*>* retval = _list;
   _list = 0;
   return retval;
}


C_grid_translation_declaration_list* C_grid_translation_declaration_list::duplicate() const
{
   return new C_grid_translation_declaration_list(*this);
}


C_grid_translation_declaration_list::~C_grid_translation_declaration_list()
{
   if (_list) {
      std::list<C_grid_translation_declaration*>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i)
         delete *i;
      delete _list;
   }
}

void C_grid_translation_declaration_list::checkChildren() 
{
   if (_list) {
      std::list<C_grid_translation_declaration*>::iterator i, begin, end;
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

void C_grid_translation_declaration_list::recursivePrint() 
{
   if (_list) {
      std::list<C_grid_translation_declaration*>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 
