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

#include "C_connection_script_definition_body.h"
#include "C_declaration.h"
#include "C_directive.h"
#include "C_connection_script_declaration.h"
#include "DataItem.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_connection_script_definition_body::internalExecute(LensContext *c)
{
   C_connection_script_declaration *current;
   std::list<C_connection_script_declaration*>::iterator i, end = _list->end();
   for (i = _list->begin(); i != end; i++) {
      current = *i;
      current->execute(c);
      if (current->isReturn()) {
         _rval = current->getRVal();
         break;
      }
   }
}


C_connection_script_definition_body::C_connection_script_definition_body(
   const C_connection_script_definition_body& rv)
   : C_production(rv), _rval(0), _tdError(0)
{
   _list = new std::list<C_connection_script_declaration*>;
   if (rv._list) {
      std::list<C_connection_script_declaration*>::iterator i, 
	 end = rv._list->end();
      for (i = rv._list->begin(); i != end; ++i) {
	 _list->push_back((*i)->duplicate());
      }
   }
   if (rv._tdError) {
      _tdError = rv._tdError->duplicate();
   }
}


C_connection_script_definition_body::C_connection_script_definition_body(
   C_connection_script_declaration *d, SyntaxError * error)
   : C_production(error), _rval(0), _tdError(0)
{
   _list = new std::list<C_connection_script_declaration*>;
   _list->push_back(d);
}


C_connection_script_definition_body::C_connection_script_definition_body(
   C_connection_script_definition_body *b, C_connection_script_declaration *d, 
   SyntaxError * error)
   : C_production(error), _rval(0), _tdError(0)
{
   if (b) {
      if (b->isError()) {
	 delete _error;
	 _error = b->_error->duplicate();
      }
      _list = b->releaseList();
      if (d) _list->push_back(d);
   }
   delete b;
}


std::list<C_connection_script_declaration*>* 
C_connection_script_definition_body::releaseList()
{
   std::list<C_connection_script_declaration*>* retval = _list;
   _list = 0;
   return retval;
}


C_connection_script_definition_body* 
C_connection_script_definition_body::duplicate() const
{
   return new C_connection_script_definition_body(*this);
}


C_connection_script_definition_body::~C_connection_script_definition_body()
{
   if (_list) {
      std::list<C_connection_script_declaration*>::iterator i, 
	 end = _list->end();
      for (i = _list->begin(); i != end; i++) {
         delete *i;
      }
   }
   delete _list;
   delete _tdError;
}

void C_connection_script_definition_body::checkChildren() 
{
   if (_list) {
      std::list<C_connection_script_declaration*>::iterator i, begin, end;
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

void C_connection_script_definition_body::recursivePrint() 
{
   if (_list) {
      std::list<C_connection_script_declaration*>::iterator i, begin, end;
      begin =_list->begin();
      end =_list->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 

void C_connection_script_definition_body::printTdError() 
{
   _tdError->setOriginal();
   _tdError->setError(true);   
   if (_tdError) {
      _tdError->printMessage();
   }
}
