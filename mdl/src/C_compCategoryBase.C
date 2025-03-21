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

#include "C_compCategoryBase.h"
#include "C_generalList.h"
#include "C_instanceMapping.h"
#include "CompCategoryBase.h"
#include "C_interfacePointerList.h"
#include "MdlContext.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "DuplicateException.h"
#include "NotFoundException.h"
#include "Interface.h"
#include "StructType.h"
#include "MemberToInterface.h"
#include "Constants.h"
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <iostream>

void C_compCategoryBase::execute(MdlContext* context) 
{
   // look at: void C_compCategoryBase::
   // executeCompCategoryBase(MdlContext* context, CompCategoryBase* cc) 
}

C_compCategoryBase::C_compCategoryBase() 
   : C_interfaceImplementorBase()
{
}

C_compCategoryBase::C_compCategoryBase(const std::string& name, 
				       C_interfacePointerList* ipl,
				       C_generalList* gl) 
   : C_interfaceImplementorBase(name, ipl, gl) 
{
}

void C_compCategoryBase::duplicate(std::unique_ptr<C_compCategoryBase>&& rv) const
{
   rv.reset(new C_compCategoryBase(*this));
}

void C_compCategoryBase::executeCompCategoryBase(
   MdlContext* context, CompCategoryBase* cc) const
{
   if (_generalList->getPhases()) {
      std::unique_ptr<std::vector<Phase*> > phases;
      _generalList->releasePhases(phases);
                                                                                   
      std::vector<Phase*>::iterator it, end = phases->end();
      for (it = phases->begin(); it != end; ++it) {
         (*it)->setPackedVariables(*cc);
      }
      cc->setInstancePhases(phases);
   }

   std::unique_ptr<StructType> inAttr;
   if (_generalList->getInAttrPSet()) {
      _generalList->releaseInAttrPSet(std::move(inAttr));
   } else {
      inAttr.reset(new StructType());
   }
   inAttr->setName(INATTRPSETNAME);
   cc->setInAttrPSet(std::move(inAttr));
   if (_generalList->getTriggeredFunctions()) { 
      std::unique_ptr<std::vector<TriggeredFunction*> > triggeredFunctions;
      _generalList->releaseTriggeredFunctions(triggeredFunctions);
      cc->setTriggeredFunctions(triggeredFunctions);
   }
}

C_compCategoryBase::~C_compCategoryBase() 
{
}


