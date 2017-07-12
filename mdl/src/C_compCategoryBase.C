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

void C_compCategoryBase::duplicate(std::auto_ptr<C_compCategoryBase>& rv) const
{
   rv.reset(new C_compCategoryBase(*this));
}

void C_compCategoryBase::executeCompCategoryBase(
   MdlContext* context, CompCategoryBase* cc) const
{
   if (_generalList->getPhases()) {
      std::auto_ptr<std::vector<Phase*> > phases;
      _generalList->releasePhases(phases);
                                                                                   
      std::vector<Phase*>::iterator it, end = phases->end();
      for (it = phases->begin(); it != end; ++it) {
         (*it)->setPackedVariables(*cc);
      }
      cc->setInstancePhases(phases);
   }

   std::auto_ptr<StructType> inAttr;
   if (_generalList->getInAttrPSet()) {
      _generalList->releaseInAttrPSet(inAttr);
   } else {
      inAttr.reset(new StructType());
   }
   inAttr->setName(INATTRPSETNAME);
   cc->setInAttrPSet(inAttr);
   if (_generalList->getTriggeredFunctions()) { 
      std::auto_ptr<std::vector<TriggeredFunction*> > triggeredFunctions;
      _generalList->releaseTriggeredFunctions(triggeredFunctions);
      cc->setTriggeredFunctions(triggeredFunctions);
   }
}

C_compCategoryBase::~C_compCategoryBase() 
{
}


