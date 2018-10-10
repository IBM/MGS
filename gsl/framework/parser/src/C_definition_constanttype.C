// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_definition_constanttype.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "Simulation.h"
#include "ConstantType.h"
#include "ConstantTypeDataItem.h"
#include "TypeRegistry.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

#include <memory>

void C_definition_constanttype::internalExecute(LensContext *c)
{
   _declarator->execute(c);

   ConstantType* tt = 
      c->sim->getConstantType(_declarator->getName());
   _instanceFactory = dynamic_cast<InstanceFactory*>(tt);
   if (_instanceFactory == 0) {
      std::string mes = 
	 "dynamic cast of ConstantType to InstanceFactory failed";
      throwError(mes);
   }
   ConstantTypeDataItem* ttdi = new ConstantTypeDataItem;
   ttdi->setConstantType(tt);
   std::unique_ptr<DataItem> diap(ttdi);
   try {
      c->symTable.addEntry(_declarator->getName(), diap);
   } catch (SyntaxErrorException& e) {
      throwError("While defining constant, " + e.getError());
   }
}

C_definition_constanttype::C_definition_constanttype(
   const C_definition_constanttype& rv)
   : C_definition(rv), _declarator(0), _instanceFactory(rv._instanceFactory)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
}

C_definition_constanttype::C_definition_constanttype(
   C_declarator *d, SyntaxError * error)
   : C_definition(error), _declarator(d), _instanceFactory(0)
{
}


C_definition_constanttype* C_definition_constanttype::duplicate() const
{
   return new C_definition_constanttype(*this);
}


std::string C_definition_constanttype::getDeclarator()
{
   return _declarator->getName();
}

InstanceFactory *C_definition_constanttype::getInstanceFactory()
{
   return _instanceFactory;
}


C_definition_constanttype::~C_definition_constanttype()
{
   delete _declarator;
}

void C_definition_constanttype::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
} 

void C_definition_constanttype::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   printErrorMessage();
} 
