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

#include "C_definition_variabletype.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "Simulation.h"
#include "VariableType.h"
#include "VariableTypeDataItem.h"
#include "TypeRegistry.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_phase_mapping_list.h"
//#include "CompCategoryBase.h"
#include "DistributableCompCategoryBase.h"   // modified by Jizhu Lu on 02/15/2006
#include "C_name.h"

#include <memory>

void C_definition_variabletype::internalExecute(LensContext *c)
{
   _declarator->execute(c);

   VariableType* tt = 
      c->sim->getVariableType(_declarator->getName());
   if (getArgument() == "global")
   {
	   tt->setCreatingInstanceAtEachMPIProcess();
   }

   _instanceFactory = dynamic_cast<InstanceFactory*>(tt);
   if (_instanceFactory == 0) {
      std::string mes = 
	 "dynamic cast of VariableType to InstanceFactory failed";
      throwError(mes);
   }

//   CompCategoryBase* ccBase = dynamic_cast<CompCategoryBase*>(tt);
   DistributableCompCategoryBase* ccBase = dynamic_cast<DistributableCompCategoryBase*>(tt);  // modified by Jizhu Lu on 02/15/2006
   if(ccBase) {
      c->setCurrentCompCategoryBase(ccBase);
      if (_phase_mapping_list) {
	 _phase_mapping_list->execute(c);
      }      
      ccBase->setUnmappedPhases(c);
#ifdef HAVE_MPI      
      ccBase->setDistributionTemplates();
#endif // HAVE_MPI
   }

   VariableTypeDataItem* ttdi = new VariableTypeDataItem;
   ttdi->setVariableType(tt);
   std::auto_ptr<DataItem> diap(ttdi);
   try {
      c->symTable.addEntry(_declarator->getName(), diap);
   } catch (SyntaxErrorException& e) {
      throwError("While defining variable, " + e.getError());
   }

}

C_definition_variabletype::C_definition_variabletype(
   const C_definition_variabletype& rv)
   : C_definition(rv), _declarator(0), _instanceFactory(rv._instanceFactory),
     _phase_mapping_list(0), _info(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._phase_mapping_list) {
      _phase_mapping_list = rv._phase_mapping_list->duplicate();
   }
   if (rv._info)
   {
	   _info = rv._info->duplicate();
   }
}

C_definition_variabletype::C_definition_variabletype(
   C_declarator *d, C_phase_mapping_list* p, SyntaxError * error)
   : C_definition(error), _declarator(d), _instanceFactory(0), 
     _phase_mapping_list(p), _info(0)
{
}

C_definition_variabletype::C_definition_variabletype(
   C_declarator *d, C_name* info, C_phase_mapping_list* p, SyntaxError * error)
   : C_definition(error), _declarator(d), _info(info), _instanceFactory(0), 
     _phase_mapping_list(p)
{
}

C_definition_variabletype* C_definition_variabletype::duplicate() const
{
   return new C_definition_variabletype(*this);
}

std::string C_definition_variabletype::getDeclarator()
{
   return _declarator->getName();
}

std::string C_definition_variabletype::getArgument()
{
	if (_info)
	{
		return _info->getName();
	}
	else return std::string("");
}

InstanceFactory *C_definition_variabletype::getInstanceFactory()
{
   return _instanceFactory;
}


C_definition_variabletype::~C_definition_variabletype()
{
   delete _declarator;
   delete _phase_mapping_list;
   delete _info;
}

void C_definition_variabletype::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_phase_mapping_list) {
      _phase_mapping_list->checkChildren();
      if (_phase_mapping_list->isError()) {
         setError();
      }
   }
   // no need to check children for _info
} 

void C_definition_variabletype::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_phase_mapping_list) {
      _phase_mapping_list->recursivePrint();
   }
   printErrorMessage();
} 
