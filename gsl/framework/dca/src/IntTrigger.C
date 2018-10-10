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

#include "IntTrigger.h"
#include "DataItem.h"
#include "IntArrayDataItem.h"
#include "NumericDataItem.h"
#include "Simulation.h"
#include "Service.h"
#include "NumericDataItem.h"
#include "StringDataItem.h"
#include "BoolDataItem.h"
#include "TriggerDataItem.h"
#include "ServiceDataItem.h"
#include "PhaseDataItem.h"
#include "Phase.h"
#include "GenericService.h"

#include <cassert>

IntTrigger::IntTrigger(Simulation& sim, std::vector<DataItem*> const & args)
   : TriggerBase(), _sim(sim)
{
   if (args.size() != 6) {
      std::cerr<<"Int trigger accepts 6 arguments!"<<std::endl;
      exit(-1);
   }
   std::vector<DataItem*>::const_iterator iter = args.begin();
   StringDataItem* descriptionDI = dynamic_cast<StringDataItem*>(*iter);
   if (descriptionDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to StringDataItem failed on IntTrigger!"<<std::endl;
      exit(-1);
   }
   _description = descriptionDI->getString();
   ++iter;
   ServiceDataItem* ptrSvcDI = dynamic_cast<ServiceDataItem*>(*iter);
   if (ptrSvcDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to ServiceDataItem failed on IntTrigger!"<<std::endl;
      exit(-1);
   }
   ++iter;
   StringDataItem* opDI = dynamic_cast<StringDataItem*>(*iter);
   if (opDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to StringDataItem failed on IntTrigger!"<<std::endl;
      exit(-1);
   }
   _op = opDI->getString();
   ++iter;
   NumericDataItem* criterionDI = dynamic_cast<NumericDataItem*>(*iter);
   if (criterionDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on IntTrigger!"<<std::endl;
      exit(-1);

   }
   _criterion = criterionDI->getInt();
   ++iter;
   NumericDataItem* delayDI = dynamic_cast<NumericDataItem*>(*iter);
   if (delayDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on IntTrigger!"<<std::endl;
      exit(-1);
   }
   _delay = delayDI->getInt();
   ++iter;
   PhaseDataItem* phaseDI = dynamic_cast<PhaseDataItem*>(*iter);
   if (phaseDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to PhaseDataItem failed on IntTrigger!"<<std::endl;
      exit(-1);
   }
   _phase = phaseDI->getPhase()->getName();

   setEvaluator(_op);
   GenericService<int>* service;
   service = dynamic_cast<GenericService<int>* >(ptrSvcDI->getService());
   assert(service != 0);
   // modification required to run in distributed computing environment -- Jizhu Lu on 03/29/2006
   assert(service != NULL);
   _service = service->getData();
   _sim.addTrigger(_phase, this);
}


bool IntTrigger::status()
{
   bool ste = false;

   ste = (*this.*_evaluator)();

   if (_delay>0) {
      if (_stateHistory.size()<=_delay) {
	 _stateHistory.push_front(ste);
	 _state = false;
      }
      else {
	 _stateHistory.pop_back();
	 _stateHistory.push_front(ste);
	 _state = _stateHistory.back();
      }
   }
   else {
      _state = ste;
   }
   return _state;
}

void IntTrigger::setEvaluator(std::string op)
{
   if (_op == ">") _evaluator = &IntTrigger::isGreaterThan;
   else if (_op == "<") _evaluator = &IntTrigger::isLessThan;
   else if (_op == "==") _evaluator = &IntTrigger::isEqual;
   else if (_op == "!=") _evaluator = &IntTrigger::isNotEqual;
   else if ((_op == ">=") || (_op == "=>")) _evaluator = &IntTrigger::isGreaterThanOrEqual;
   else if ((_op == "<=") || (_op == "=<")) _evaluator = &IntTrigger::isLessThanOrEqual;
   else if (_op == "!%") _evaluator = &IntTrigger::isModulusZero;
   else if (_op == "%") _evaluator = &IntTrigger::isModulusNonZero;
   else {
      std::cerr<<"Operation not supported on IntTrigger!"<<std::endl;
      exit(-1);
   }
}


bool IntTrigger::isEqual()
{
   return *_service == _criterion;
}


bool IntTrigger::isGreaterThan()
{
   return (*_service) >_criterion;
}


bool IntTrigger::isLessThan()
{
   return (*_service) < _criterion;
}


bool IntTrigger::isNotEqual()
{
   return (*_service) != _criterion;
}


bool IntTrigger::isGreaterThanOrEqual()
{
   return (*_service) >= _criterion;
}


bool IntTrigger::isLessThanOrEqual()
{
   return (*_service) <= _criterion;
}


bool IntTrigger::isModulusZero()
{
   return !((*_service) % _criterion);
}


bool IntTrigger::isModulusNonZero()
{
   return (*_service) % _criterion;
}

void IntTrigger::duplicate(std::unique_ptr<Trigger>& dup) const
{
   dup.reset(new IntTrigger(*this));
}

IntTrigger::~IntTrigger()
{
}
