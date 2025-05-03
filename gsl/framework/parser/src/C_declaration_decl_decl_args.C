// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_decl_decl_args.h"
#include "GslContext.h"
#include "C_declarator.h"
#include "C_argument_list.h"
#include "C_ndpair_clause_list.h"
#include "DataItem.h"
#include "InstanceFactory.h"
#include "InstanceFactoryDataItem.h"
#include "FunctorDataItem.h"
#include <memory>
#include <typeinfo>
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "ArrayException.h"

std::string C_declaration_decl_decl_args::getName()
{
   return _nameDecl->getName();
}

void C_declaration_decl_decl_args::internalExecute(GslContext *c)
{
   _userDefinedType->execute(c);
   _nameDecl->execute(c);
   if (_argList) {
      _argList->execute(c);
   }
   if (_ndpair_clause_list) {
      _ndpair_clause_list->execute(c);
   }

   DataItem const *di = c->symTable.getEntry(_userDefinedType->getName());
   if (di ==0 ) {
      std::string mes = _userDefinedType->getName() + " was not found";
      throwError(mes);
   }
   InstanceFactoryDataItem const *ifdi = 
      dynamic_cast<InstanceFactoryDataItem const*>(di);
   if (ifdi ==0) {
      std::string mes = _userDefinedType->getName() + 
	 " is not a InstanceFactory";
      throwError(mes);
   }
   InstanceFactory* ifc = ifdi->getInstanceFactory();
   std::unique_ptr<DataItem> inst_ap;
   try {
      if (_argList) {
	 ifc->getInstance(inst_ap, _argList->getVectorDataItem(), c);
      } else {
	 ifc->getInstance(inst_ap, _ndpair_clause_list->getNDPList(), c);
      }
   } catch (ArrayException& e) {
      throwError(e.getError());
   } catch (SyntaxErrorException& e) {
      throwError(e.getError());
   }
   try {
      c->symTable.addEntry(_nameDecl->getName(), inst_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring argument, " + e.getError());
   }
}


C_declaration_decl_decl_args::C_declaration_decl_decl_args(
   const C_declaration_decl_decl_args& rv)
   : C_declaration(rv), _userDefinedType(0), _nameDecl(0), _argList(0),
     _ndpair_clause_list(0), _dataitem(0)
{
   if (rv._userDefinedType) {
      _userDefinedType = rv._userDefinedType->duplicate();
   }
   if (rv._nameDecl) {
      _nameDecl = rv._nameDecl->duplicate();
   }
   if (rv._argList) {
      _argList = rv._argList->duplicate();
   }
   if (rv._ndpair_clause_list) {
      _ndpair_clause_list = rv._ndpair_clause_list->duplicate();
   }
   if (rv._dataitem) {
      std::unique_ptr<DataItem> di_ap;
      rv._dataitem->duplicate(di_ap);
      _dataitem = di_ap.release();
   }
}


C_declaration_decl_decl_args::C_declaration_decl_decl_args(
   C_declarator *cat, C_declarator *name, C_argument_list *a, 
   SyntaxError * error)
   : C_declaration(error), _userDefinedType(cat), _nameDecl(name), _argList(a),
     _ndpair_clause_list(0), _dataitem(0)
{
}

C_declaration_decl_decl_args::C_declaration_decl_decl_args (
   C_ndpair_clause_list *n, C_declarator *cat, C_declarator *name, 
   SyntaxError *error)
   : C_declaration(error), _userDefinedType(cat), _nameDecl(name), _argList(0),
     _ndpair_clause_list(n), _dataitem(0)
{
}

C_declaration_decl_decl_args* C_declaration_decl_decl_args::duplicate() const
{
   return new C_declaration_decl_decl_args(*this);
}


C_declaration_decl_decl_args::~C_declaration_decl_decl_args()
{
   delete _userDefinedType;
   delete _nameDecl;
   delete _argList;
   delete _ndpair_clause_list;
   delete _dataitem;
}

void C_declaration_decl_decl_args::checkChildren() 
{
   if (_userDefinedType) {
      _userDefinedType->checkChildren();
      if (_userDefinedType->isError()) {
         setError();
      }
   }
   if (_nameDecl) {
      _nameDecl->checkChildren();
      if (_nameDecl->isError()) {
         setError();
      }
   }
   if (_argList) {
      _argList->checkChildren();
      if (_argList->isError()) {
         setError();
      }
   }
   if (_ndpair_clause_list) {
      _ndpair_clause_list->checkChildren();
      if (_ndpair_clause_list->isError()) {
         setError();
      }
   }
} 

void C_declaration_decl_decl_args::recursivePrint() 
{
   if (_userDefinedType) {
      _userDefinedType->recursivePrint();
   }
   if (_nameDecl) {
      _nameDecl->recursivePrint();
   }
   if (_argList) {
      _argList->recursivePrint();
   }
   if (_ndpair_clause_list) {
      _ndpair_clause_list->recursivePrint();
   }
   printErrorMessage();
} 
