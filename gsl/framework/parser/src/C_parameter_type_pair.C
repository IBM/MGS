// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_parameter_type_pair.h"
#include "C_declarator.h"
#include "C_init_attr_type_node.h"
#include "C_init_attr_type_edge.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_parameter_type_pair::internalExecute(GslContext *c)
{

   _declarator->execute(c);
   if (_modelType==_EDGE) {
      if(_iat_edge) _iat_edge->execute(c);
      _parameterType = _INIT;
   }
   else {
      if(_iat_node) _iat_node->execute(c);
      switch (_iat_node->getType()) {
         case C_init_attr_type_node::_NODEINIT: _parameterType = _INIT; break;
         case C_init_attr_type_node::_IN: _parameterType = _IN; break;
         case C_init_attr_type_node::_OUT: _parameterType = _OUT; break;
      }
   }
}


C_parameter_type_pair::C_parameter_type_pair(const C_parameter_type_pair& rv)
   : C_production(rv), _declarator(0), _iat_node(0), _iat_edge(0), 
     _modelType(rv._modelType), _parameterType(rv._parameterType)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._iat_edge) {
      _iat_edge = rv._iat_edge->duplicate();
   }
   if (rv._iat_node) {
      _iat_node = rv._iat_node->duplicate();
   }
}


C_parameter_type_pair::C_parameter_type_pair(
   C_declarator *d, C_init_attr_type_node *ia, SyntaxError * error)
   : C_production(error), _declarator(d), _iat_node(ia), _iat_edge(0), 
     _modelType(_NODE), _parameterType(_UNINITIALIZED)
{
}


C_parameter_type_pair::C_parameter_type_pair(
   C_declarator *d, C_init_attr_type_edge *ia, SyntaxError * error)
   : C_production(error), _declarator(d), _iat_node(0), _iat_edge(ia), 
     _modelType(_EDGE), _parameterType(_UNINITIALIZED)
{
}

C_parameter_type_pair::C_parameter_type_pair(SyntaxError * error)
   : C_production(error), _declarator(0), _iat_node(0), _iat_edge(0), 
     _modelType(_EDGE), _parameterType(_UNINITIALIZED)
{
}


std::string const & C_parameter_type_pair::getModelName() const
{
   return _declarator->getName();
}


C_parameter_type_pair* C_parameter_type_pair::duplicate() const
{
   return new C_parameter_type_pair(*this);
}


C_parameter_type_pair::~C_parameter_type_pair()
{
   delete _declarator;
   delete _iat_node;
   delete _iat_edge;
}

void C_parameter_type_pair::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_iat_node) {
      _iat_node->checkChildren();
      if (_iat_node->isError()) {
         setError();
      }
   }
   if (_iat_edge) {
      _iat_edge->checkChildren();
      if (_iat_edge->isError()) {
         setError();
      }
   }
} 

void C_parameter_type_pair::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_iat_node) {
      _iat_node->recursivePrint();
   }
   if (_iat_edge) {
      _iat_edge->recursivePrint();
   }
   printErrorMessage();
} 
