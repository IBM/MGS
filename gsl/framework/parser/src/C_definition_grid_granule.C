
// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================#include "C_definition_grid_granule.h"
#include "LensContext.h"
#include "C_parameter_type.h"
#include "C_parameter_type_list.h"
#include "C_declarator.h"
#include "Simulation.h"
#include "GranuleMapperType.h"
#include "GranuleMapperTypeDataItem.h"
#include "TypeRegistry.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

#include <memory>

void C_definition_grid_granule::internalExecute(LensContext *c)
{
   // call executes
   _declarator->execute(c);
   _constructor_ptl->execute(c);

   GranuleMapperType* gt = c->sim->getGranuleMapperType(_declarator->getName());
   _instanceFactory = dynamic_cast<InstanceFactory*>(gt);
   if (_instanceFactory == 0) {
      std::string mes = 
	 "dynamic cast of GranuleMapperType to InstanceFactory failed";
      throwError(mes);
   }
   _constructor_list = _constructor_ptl->getList();
   GranuleMapperTypeDataItem* gtdi = new GranuleMapperTypeDataItem;
   gtdi->setGranuleMapperType(gt);
   std::unique_ptr<DataItem> diap(gtdi);
   try {
      c->symTable.addEntry(_declarator->getName(), diap);
   } catch (SyntaxErrorException& e) {
      throwError("While defining grid_granule, " + e.getError());
   }
}

C_definition_grid_granule::C_definition_grid_granule(const C_definition_grid_granule& rv)
   : C_definition(rv), _declarator(0), _constructor_ptl(0), 
     _instanceFactory(rv._instanceFactory), _constructor_list(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._constructor_ptl) {
      _constructor_ptl = rv._constructor_ptl->duplicate();
   }
   if (rv._constructor_list) {
      _constructor_list = rv._constructor_list;
   }
}

C_definition_grid_granule::C_definition_grid_granule(
   C_declarator *d, C_parameter_type_list *p, SyntaxError * error)
   : C_definition(error), _declarator(d), _constructor_ptl(p), 
     _instanceFactory(0), _constructor_list(0)
{
}

C_definition_grid_granule* C_definition_grid_granule::duplicate() const
{
   return new C_definition_grid_granule(*this);
}

std::string C_definition_grid_granule::getDeclarator()
{
   return _declarator->getName();
}

std::list<C_parameter_type>* C_definition_grid_granule::getConstructorParams()
{
   return _constructor_list;
}

InstanceFactory *C_definition_grid_granule::getInstanceFactory()
{
   return _instanceFactory;
}

C_definition_grid_granule::~C_definition_grid_granule()
{
   delete _constructor_ptl;
   delete _declarator;
}

void C_definition_grid_granule::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_constructor_ptl) {
      _constructor_ptl->checkChildren();
      if (_constructor_ptl->isError()) {
         setError();
      }
   }
} 

void C_definition_grid_granule::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_constructor_ptl) {
      _constructor_ptl->recursivePrint();
   }
   printErrorMessage();
} 
