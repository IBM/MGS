// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_connection_script_definition.h"
#include "C_connection_script_definition_body.h"
#include "C_declarator.h"
#include "C_parameter_type_list.h"
#include "ConnectionScriptFunctor.h"
#include <memory>

void C_connection_script_definition::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   _param_type_list->execute(c);
   //_script_body->execute(c);
   _functor = new ConnectionScriptFunctor(
      _script_body, _param_type_list->getList());
   _scriptName = _declarator->getName();
}


C_connection_script_definition::C_connection_script_definition(
   const C_connection_script_definition& rv)
   : C_production(rv), _declarator(0), _param_type_list(0), 
     _script_body(0), _functor(0), _scriptName(rv._scriptName)
{
   if (rv._functor) {
      std::auto_ptr<Functor> fap;
      rv._functor->duplicate(fap);
      _functor = fap.release();
   }
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._param_type_list) {
      _param_type_list = rv._param_type_list->duplicate();
   }
   if (rv._script_body) {
      _script_body = rv._script_body->duplicate();
   }
}


C_connection_script_definition::C_connection_script_definition (
   C_declarator *d, C_parameter_type_list *p, 
   C_connection_script_definition_body *c, SyntaxError * error)
   : C_production(error), _declarator(d), _param_type_list(p), 
     _script_body(c), _functor(0), _scriptName("Undefined!")
{
}


C_connection_script_definition* 
C_connection_script_definition::duplicate() const
{
   return new C_connection_script_definition(*this);
}

std::string const & C_connection_script_definition::getName() 
{
   if (_declarator) {
      return _declarator->getName();
   } else {
      _name = "__Empty__";
      return _name;
   }
}

std::list<C_parameter_type>* 
C_connection_script_definition::getFunctionParameters()
{
   return _param_type_list->getList();
}


C_connection_script_definition::~C_connection_script_definition()
{
   delete _declarator;
   delete _param_type_list;
   delete _script_body;
   delete _functor;
}

void C_connection_script_definition::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_param_type_list) {
      _param_type_list->checkChildren();
      if (_param_type_list->isError()) {
         setError();
      }
   }
   if (_script_body) {
      _script_body->checkChildren();
      if (_script_body->isError()) {
         setError();
      }
   }
} 

void C_connection_script_definition::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_param_type_list) {
      _param_type_list->recursivePrint();
   }
   if (_script_body) {
      _script_body->recursivePrint();
   }
   printErrorMessage();
} 
