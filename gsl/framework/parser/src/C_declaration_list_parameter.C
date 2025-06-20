// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_list_parameter.h"
#include "C_type_specifier.h"
#include "C_initializable_type_specifier.h"
#include "C_declarator.h"
#include "C_argument_list.h"
#include "GslContext.h"
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "SyntaxError.h"
#include "ArgumentListHelper.h"
#include "SyntaxErrorException.h"

void C_declaration_list_parameter::internalExecute(GslContext *c)
{
   if (_typeSpec)
      _typeSpec->execute(c);
   _declarator->execute(c);
   _argumentList->execute(c);
   std::unique_ptr<DataItem> diap;
   ArgumentListHelper helper;
   helper.getDataItem(diap, c, _argumentList, _typeSpec);
   try {
      c->symTable.addEntry(_declarator->getName(), diap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring list parameter, " + e.getError());
   }
}


C_declaration_list_parameter* C_declaration_list_parameter::duplicate() const
{
   return new C_declaration_list_parameter(*this);
}


C_declaration_list_parameter::C_declaration_list_parameter(
   C_declarator *d, C_argument_list *a, SyntaxError * error)
   : C_declaration(error), _typeSpec(0), _declarator(d), _argumentList(a)
{
}


C_declaration_list_parameter::C_declaration_list_parameter(
   C_type_specifier *t, C_declarator *d, C_argument_list *a, 
   SyntaxError * error)
   : C_declaration(error), _typeSpec(t), _declarator(d), _argumentList(a)
{
}


C_declaration_list_parameter::C_declaration_list_parameter(
   const C_declaration_list_parameter& rv)
   : C_declaration(rv), _typeSpec(0), _declarator(0), _argumentList(0)
{
   if (rv._typeSpec) {
      _typeSpec = rv._typeSpec->duplicate();
   }
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._argumentList) {
      _argumentList = rv._argumentList->duplicate();
   }
}


C_declaration_list_parameter::~C_declaration_list_parameter()
{
   delete _typeSpec;
   delete _declarator;
   delete _argumentList;
}

void C_declaration_list_parameter::checkChildren() 
{
   if (_typeSpec) {
      _typeSpec->checkChildren();
      if (_typeSpec->isError()) {
         setError();
      }
   }
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_argumentList) {
      _argumentList->checkChildren();
      if (_argumentList->isError()) {
         setError();
      }
   }
} 

void C_declaration_list_parameter::recursivePrint() 
{
   if (_typeSpec) {
      _typeSpec->recursivePrint();
   }
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_argumentList) {
      _argumentList->recursivePrint();
   }
   printErrorMessage();
} 
