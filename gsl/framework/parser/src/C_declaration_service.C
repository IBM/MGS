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

#include "C_declaration_service.h"
#include "Service.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "C_service.h"
#include "ServiceDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_service::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   _service->execute(c);

   ServiceDataItem *sdi = new ServiceDataItem;
   std::auto_ptr<DataItem> sdi_ap(sdi);

   Service *ns = _service->getService();;
   sdi->setService(ns);

   try {
      c->symTable.addEntry(_declarator->getName(), sdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring service, " + e.getError());
   }
}


C_declaration_service* C_declaration_service::duplicate() const
{
   return new C_declaration_service(*this);
}


C_declaration_service::C_declaration_service(
   C_declarator *d, C_service *s, SyntaxError * error)
   : C_declaration(error), _declarator(d), _service(s)
{
}


C_declaration_service::C_declaration_service(const C_declaration_service& rv)
   : C_declaration(rv), _declarator(0), _service(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._service) {
      _service = rv._service->duplicate();
   }
}


C_declaration_service::~C_declaration_service()
{
   delete _declarator;
   delete _service;
}

void C_declaration_service::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_service) {
      _service->checkChildren();
      if (_service->isError()) {
         setError();
      }
   }
} 

void C_declaration_service::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_service) {
      _service->recursivePrint();
   }
   printErrorMessage();
} 
