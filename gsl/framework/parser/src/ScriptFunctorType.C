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

#include "ScriptFunctorType.h"
#include "C_connection_script_definition.h"
#include "ConnectorFunctor.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

ScriptFunctorType::ScriptFunctorType(C_connection_script_definition *def, std::string const &name)
:_functorName(name)
{
   _c_script_def = def->duplicate();

}


void ScriptFunctorType::getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup)
{
   FunctorDataItem* fdi = new FunctorDataItem;
   fdi->setFunctor(_c_script_def->getFunctor());
   std::unique_ptr<DataItem> apdi(fdi);

   DataItemQueriable* diq = new DataItemQueriable(apdi);
   diq->setName(getName());
   diq->setDescription(getDescription());
   std::unique_ptr<DataItemQueriable> apq(diq);

   dup.reset(new InstanceFactoryQueriable(this));
   dup->addQueriable(apq);
   dup->setName(getName());
}


Functor* ScriptFunctorType::getFunctor()
{
   return _c_script_def->getFunctor();
}


ScriptFunctorType::ScriptFunctorType(ScriptFunctorType const *sft)
:_functorName(sft->_functorName)
{
   _c_script_def = sft->_c_script_def->duplicate();
}


void ScriptFunctorType::getFunctor(std::unique_ptr<Functor> & r_aptr)
{
   _c_script_def->getFunctor()->duplicate(r_aptr);
}


std::string ScriptFunctorType::getName()
{
   return _functorName;
}


std::string ScriptFunctorType::getDescription()
{
   return std::string(ConnectorFunctor::_category);
}


ScriptFunctorType::~ScriptFunctorType()
{
   delete _c_script_def;
}
