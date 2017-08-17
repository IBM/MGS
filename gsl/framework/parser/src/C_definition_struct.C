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

#include "C_definition_struct.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "Simulation.h"
#include "StructType.h"
#include "StructTypeDataItem.h"
#include "TypeRegistry.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

#include <memory>

void C_definition_struct::internalExecute(LensContext *c)
{
   _declarator->execute(c);

   StructType* tt = c->sim->getStructType(_declarator->getName());
   _instanceFactory = dynamic_cast<InstanceFactory*>(tt);
   if (_instanceFactory == 0) {
      std::string mes = "dynamic cast of StructType to InstanceFactory failed";
      throwError(mes);
   }
   StructTypeDataItem* ttdi = new StructTypeDataItem;
   ttdi->setStructType(tt);
   std::auto_ptr<DataItem> diap(ttdi);
   try {
      c->symTable.addEntry(_declarator->getName(), diap);
   } catch (SyntaxErrorException& e) {
      throwError("While defining struct, " + e.getError());
   }
}

C_definition_struct::C_definition_struct(const C_definition_struct& rv)
   : C_definition(rv), _declarator(0), _instanceFactory(rv._instanceFactory)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
}

C_definition_struct::C_definition_struct(C_declarator *d, SyntaxError * error)
   : C_definition(error), _declarator(d), _instanceFactory(0)
{
}

C_definition_struct* C_definition_struct::duplicate() const
{
   return new C_definition_struct(*this);
}

std::string C_definition_struct::getDeclarator()
{
   return _declarator->getName();
}

InstanceFactory *C_definition_struct::getInstanceFactory()
{
   return _instanceFactory;
}

C_definition_struct::~C_definition_struct()
{
   delete _declarator;
}

void C_definition_struct::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
} 

void C_definition_struct::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   printErrorMessage();
} 
