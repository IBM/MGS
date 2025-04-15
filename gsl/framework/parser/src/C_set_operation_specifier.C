// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_set_operation_specifier.h"
#include "C_set_operation.h"
#include "C_declarator.h"
#include "C_argument_list.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_set_operation_specifier::internalExecute(LensContext *c)
{
   _dcl->execute(c);
   _sop->execute(c);
   _fdl->execute(c);
   _arl->execute(c);
}


C_set_operation_specifier::C_set_operation_specifier(
   const C_set_operation_specifier& rv)
   : C_production(rv), _dcl(0), _sop(0), _id(rv._id), _fdl(0), _arl(0)
{
   if (rv._dcl) {
      _dcl = rv._dcl->duplicate();
   }
   if (rv._sop) {
      _sop = rv._sop->duplicate();
   }
   if (rv._fdl) {
      _fdl = rv._fdl->duplicate();
   }
   if (rv._arl) {
      _arl = rv._arl->duplicate();
   }
}


C_set_operation_specifier::C_set_operation_specifier(
   C_declarator *dcl, C_set_operation *sop, std::string id, C_declarator *fdl, 
   C_argument_list *arl, SyntaxError * error)
   : C_production(error), _dcl(dcl), _sop(sop), _id(id), _fdl(fdl), _arl(arl)
{
}


C_set_operation_specifier* C_set_operation_specifier::duplicate() const
{
   return new C_set_operation_specifier(*this);
}


C_set_operation_specifier::~C_set_operation_specifier()
{
   delete _dcl;
   delete _sop;
   delete _fdl;
   delete _arl;
}

void C_set_operation_specifier::checkChildren() 
{
   if (_dcl) {
      _dcl->checkChildren();
      if (_dcl->isError()) {
         setError();
      }
   }
   if (_sop) {
      _sop->checkChildren();
      if (_sop->isError()) {
         setError();
      }
   }
   if (_fdl) {
      _fdl->checkChildren();
      if (_fdl->isError()) {
         setError();
      }
   }
   if (_arl) {
      _arl->checkChildren();
      if (_arl->isError()) {
         setError();
      }
   }
} 

void C_set_operation_specifier::recursivePrint() 
{
   if (_dcl) {
      _dcl->recursivePrint();
   }
   if (_sop) {
      _sop->recursivePrint();
   }
   if (_fdl) {
      _fdl->recursivePrint();
   }
   if (_arl) {
      _arl->recursivePrint();
   }
   printErrorMessage();
} 
