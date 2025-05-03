// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_publisher.h"
#include "Publisher.h"
#include "GslContext.h"
#include "C_declarator.h"
#include "C_query_path.h"
#include "PublisherDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_publisher::internalExecute(GslContext *c)
{
   _declarator->execute(c);
   _query_path->execute(c);

   PublisherDataItem *pdi = new PublisherDataItem;
   std::unique_ptr<DataItem> pdi_ap(pdi);

   Publisher *ns = _query_path->getPublisher();;
   pdi->setPublisher(ns);

   try {
      c->symTable.addEntry(_declarator->getName(), pdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring publisher, " + e.getError());
   }
}


C_declaration_publisher* C_declaration_publisher::duplicate() const
{
   return new C_declaration_publisher(*this);
}


C_declaration_publisher::C_declaration_publisher(
   C_declarator *d, C_query_path *n, SyntaxError * error)
   : C_declaration(error), _declarator(d), _query_path(n)
{
}


C_declaration_publisher::C_declaration_publisher(
   const C_declaration_publisher& rv)
   : C_declaration(rv), _declarator(0), _query_path(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._query_path) {
      _query_path = rv._query_path->duplicate();
   }
}


C_declaration_publisher::~C_declaration_publisher()
{
   delete _declarator;
   delete _query_path;
}

void C_declaration_publisher::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_query_path) {
      _query_path->checkChildren();
      if (_query_path->isError()) {
         setError();
      }
   }
} 

void C_declaration_publisher::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_query_path) {
      _query_path->recursivePrint();
   }
   printErrorMessage();
} 
