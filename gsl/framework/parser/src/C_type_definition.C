// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_type_definition.h"
#include "C_declarator.h"
#include "C_grid_definition_body.h"
#include "C_composite_definition_body.h"
#include "LensContext.h"
#include "RepertoireFactoryDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"


void C_type_definition::internalExecute(LensContext *c)
{
   _typeName->execute(c);
   RepertoireFactoryDataItem * rdi = new RepertoireFactoryDataItem;
   std::unique_ptr<DataItem> di_ap(rdi);

   if(_type == _GRID) {
      // here i have to transfer the parse tree
      // do not "execute" it, that will be deferred
      // for later. i'm simply transfering ownership
      // of _gridDefBody to factory data item.

      _gridDefBody->execute(c);
      std::unique_ptr<RepertoireFactory> dup(
	 new C_grid_definition_body(*_gridDefBody));
      rdi->setFactory(dup);
      try {
	 c->symTable.addEntry(_typeName->getName(), di_ap);
      } catch (SyntaxErrorException& e) {
	 throwError("While adding type definition, " + e.getError());
      }

      // instance: create GridRepertoire (to be stored in symbol table)
      if(_declaration) {
         _instanceName->execute(c);
         const RepertoireFactoryDataItem* rfdi;
         rfdi  = dynamic_cast<const RepertoireFactoryDataItem*>(
	    c->symTable.getEntry(_typeName->getName()));
         if (rfdi == 0) {
	    std::string mes = 
	       "dynamic cast of DataItem to RepertoireFactoryDataItem failed";
	    throwError(mes);
         }
         rfdi->getFactory()->createRepertoire(_instanceName->getName(), c);
      }
   }

   if(_type == _COMPOSITE) {
      _compositeDefBody->execute(c);
      std::unique_ptr<RepertoireFactory> dup(
	 new C_composite_definition_body(*_compositeDefBody));
      rdi->setFactory(dup);
      try {
	 c->symTable.addEntry(_typeName->getName(), di_ap);
      } catch (SyntaxErrorException& e) {
	 throwError("While adding type definition composite, " + e.getError());
      }

      if(_declaration) {
         _instanceName->execute(c);
         const RepertoireFactoryDataItem* rfdi;
         rfdi  = dynamic_cast<const RepertoireFactoryDataItem*>(
	    c->symTable.getEntry(_typeName->getName()));
         if (rfdi == 0) {
	    std::string mes = 
	       "dynamic cast of DataItem to RepertoireFactoryDataItem failed";
	    throwError(mes);
         }
         rfdi->getFactory()->createRepertoire(_instanceName->getName(), c);
      }
   }
}


C_type_definition::C_type_definition(const C_type_definition& rv)
   : C_production(rv), _declaration(rv._declaration),
     _typeName(0), _instanceName(0), _gridDefBody(0), _compositeDefBody(0), 
     _type(rv._type)
{
   if (rv._typeName) {
      _typeName = rv._typeName->duplicate();
   }
   if (rv._instanceName) {
      _instanceName = rv._instanceName->duplicate();
   }
   if (rv._gridDefBody) {
      _gridDefBody = dynamic_cast<C_grid_definition_body*>(
	 rv._gridDefBody->duplicate());
   }
   if (rv._compositeDefBody) {
      _compositeDefBody = dynamic_cast<C_composite_definition_body*>(
	 rv._compositeDefBody->duplicate());
   }
}


C_type_definition::C_type_definition(
   C_declarator *d, C_grid_definition_body *g, SyntaxError * error)
   : C_production(error), _declaration(false), _typeName(d), _instanceName(0), 
     _gridDefBody(g), _compositeDefBody(0), _type(_GRID)
{
}


C_type_definition::C_type_definition(
   C_declarator *d1, C_grid_definition_body *g, C_declarator *d2, 
   SyntaxError * error)
   : C_production(error), _declaration(true), _typeName(d1), _instanceName(d2),
     _gridDefBody(g), _compositeDefBody(0), _type(_GRID)
{
}


C_type_definition::C_type_definition(
   C_declarator *d, C_composite_definition_body *c, SyntaxError * error)
   : C_production(error), _declaration(false), _typeName(d), _instanceName(0),
     _gridDefBody(0), _compositeDefBody(c), _type(_COMPOSITE)
{
}


C_type_definition::C_type_definition(
   C_declarator *d1, C_composite_definition_body *c, C_declarator *d2, 
   SyntaxError * error)
   : C_production(error), _declaration(true), _typeName(d1), _instanceName(d2),
     _gridDefBody(0), _compositeDefBody(c), _type(_COMPOSITE)
{
}

C_type_definition::C_type_definition(SyntaxError * error)
   : C_production(error), _declaration(false), _typeName(0), _instanceName(0), 
     _gridDefBody(0), _compositeDefBody(0), _type(_GRID)
{
}

C_type_definition* C_type_definition::duplicate() const
{
   return new C_type_definition(*this);
}


C_type_definition::~C_type_definition()
{
   delete _typeName;
   delete _instanceName;
   delete _gridDefBody;
   delete _compositeDefBody;
}

void C_type_definition::checkChildren() 
{
   if (_typeName) {
      _typeName->checkChildren();
      if (_typeName->isError()) {
         setError();
      }
   }
   if (_instanceName) {
      _instanceName->checkChildren();
      if (_instanceName->isError()) {
         setError();
      }
   }
   if (_gridDefBody) {
      _gridDefBody->checkChildren();
      if (_gridDefBody->isError()) {
         setError();
      }
   }
   if (_compositeDefBody) {
      _compositeDefBody->checkChildren();
      if (_compositeDefBody->isError()) {
         setError();
      }
   }
} 

void C_type_definition::recursivePrint() 
{
   if (_typeName) {
      _typeName->recursivePrint();
   }
   if (_instanceName) {
      _instanceName->recursivePrint();
   }
   if (_gridDefBody) {
      _gridDefBody->recursivePrint();
   }
   if (_compositeDefBody) {
      _compositeDefBody->recursivePrint();
   }
   printErrorMessage();
} 
