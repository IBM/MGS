// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "CompositeTriggerServiceTrigger.h"
#include "Simulation.h"
#include "StringDataItem.h"
#include "BoolDataItem.h"
#include "TriggerDataItem.h"
#include "PhaseDataItem.h"
#include "Phase.h"
#include <iostream>
#include "ServiceDataItem.h"
#include "Service.h"
#include "GenericService.h"

// default criterionA & criterionB is true
CompositeTriggerServiceTrigger::CompositeTriggerServiceTrigger(Simulation& sim, 
				   std::vector<DataItem*> const & args)
   : _sim(sim)
{
   if ((args.size() != 5) && (args.size() != 8)) {
      std::cerr<<"Composite trigger accepts 5 or 8 arguments!"<<std::endl;
      exit(-1);
   }
   std::vector<DataItem*>::const_iterator iter = args.begin();
   StringDataItem* descriptionDI = dynamic_cast<StringDataItem*>(*iter);
   if (descriptionDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to StringDataItem failed on CompositeTriggerServiceTriggerServiceTrigger!"<<std::endl;
      exit(-1);
   }
   _description = descriptionDI->getString();
   ++iter;
   TriggerDataItem* triggerADI = dynamic_cast<TriggerDataItem*>(*iter);
   if (triggerADI == 0) {
      std::cerr<<"Dynamic cast of DataItem to TriggerDataItem failed on CompositeTriggerServiceTriggerServiceTrigger!"<<std::endl;
      exit(-1);
   }
   _triggerA = triggerADI->getTrigger();
   ++iter;
   NumericDataItem* criterionADI = dynamic_cast<NumericDataItem*>(*iter);
   if (criterionADI == 0) {
      std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on CompositeTriggerServiceTriggerServiceTrigger!"<<std::endl;
      exit(-1);
   }
   _criterionA = criterionADI->getBool();
   ++iter;

   if (args.size() == 8) {
      StringDataItem* opDI = dynamic_cast<StringDataItem*>(*iter);
      if (opDI == 0) {
	 std::cerr<<"Dynamic cast of DataItem to StringDataItem failed on CompositeTriggerServiceTriggerServiceTrigger!"<<std::endl;
	 exit(-1);
      }
      _op = opDI->getString();
      ++iter;
      ServiceDataItem* ptrSvcDI = dynamic_cast<ServiceDataItem*>(*iter);
      if (ptrSvcDI == 0) {
	 std::cerr<<"Dynamic cast of DataItem to ServiceDataItem failed on CompositeTriggerServiceTriggerServiceTrigger!"<<std::endl;
	 exit(-1);
      }
      GenericService<unsigned>* service;
      service = dynamic_cast<GenericService<unsigned>*>(ptrSvcDI->getService());
      assert(service != 0);
      // modification required to run in distributed computing environment -- from Jizhu
      // Lu on 03/29/2006
      _serviceB = service->getData();
      ++iter;
      NumericDataItem* criterionBDI = dynamic_cast<NumericDataItem*>(*iter);
      if (criterionBDI == 0) {
	 std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on CompositeTriggerServiceTriggerServiceTrigger!"<<std::endl;
	 exit(-1);
      }
      _criterionB = criterionBDI->getBool();
      ++iter;
   } else {
      _op = "single";
      _serviceB = 0;
      _criterionB = 1;
   }

   NumericDataItem* delayDI = dynamic_cast<NumericDataItem*>(*iter);
   if (delayDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on CompositeTriggerServiceTriggerServiceTrigger!"<<std::endl;
      exit(-1);
   }
   _delay = delayDI->getInt();
   ++iter;

   PhaseDataItem* phaseDI = dynamic_cast<PhaseDataItem*>(*iter);
   if (phaseDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to PhaseDataItem failed on CompositeTriggerServiceTriggerServiceTrigger!"<<std::endl;
      exit(-1);
   }
   _phase = phaseDI->getPhase()->getName();

   setEvaluator();
   _sim.addTrigger(_phase, this);
}

CompositeTriggerServiceTrigger::CompositeTriggerServiceTrigger(Simulation& sim, Trigger* t1, unsigned* t2, 
				   std::string& op)
   : _sim(sim), _triggerA(t1), _serviceB(t2), _op(op), _criterionA(1), 
     _criterionB(1)
{
   _description = "Auto Generated Composite Trigger";
   _delay= 0;
   setEvaluator();
   if ((t1 == 0) || (t2 == 0)) {
      std::cerr << "Warning: CompositeTriggerServiceTriggerServiceTrigger is constructed with a NULL pointer" << std::endl;
   }
   //_phase = _sim.findLaterPhase(t1->getPhase(), t2->getPhase());
   _phase = t1->getPhase();
   _sim.addTrigger(_phase, this);
}

CompositeTriggerServiceTrigger::CompositeTriggerServiceTrigger(Simulation& sim, Trigger* t1)
   : _sim(sim), _triggerA(t1), _serviceB(0), _op("single"), 
     _criterionA(1), _criterionB(1)
{
   _description = "Auto Generated Composite Trigger";
   _delay= 0;
   setEvaluator();
   if (t1 == 0) {
      std::cerr << "Warning: CompositeTriggerServiceTrigger is constructed with a NULL pointer" << std::endl;
   }
   _phase = t1->getPhase();
   _sim.addTrigger(_phase, this);
}

bool CompositeTriggerServiceTrigger::status()
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

void CompositeTriggerServiceTrigger::setEvaluator()
{
   if ((_op == "&&") || (_op == "AND") || (_op == "and")) {
      _evaluator = &CompositeTriggerServiceTrigger::isAnd;
   }
   else if ((_op == "||") || (_op == "OR") || (_op == "or")) {
      _evaluator = &CompositeTriggerServiceTrigger::isOr;
   }
   else if ((_op == "XOR") || (_op == "xor")) {
      _evaluator = &CompositeTriggerServiceTrigger::isXor;
   }
   else if ((_op == "SINGLE") || (_op == "single")) {
      _evaluator = &CompositeTriggerServiceTrigger::isSingle;
   }
   else {
      std::cerr<<"Operation not supported on CompositeTriggerServiceTriggerServiceTrigger!"<<std::endl;
      exit(-1);
   }
}


bool CompositeTriggerServiceTrigger::isAnd()
{
   return ( (_triggerA->status() == _criterionA) && 
	    ((int)(*_serviceB) == _criterionB) );
}


bool CompositeTriggerServiceTrigger::isOr()
{
   return ( (_triggerA->status() == _criterionA) || 
	    ((int)(*_serviceB) == _criterionB) );
}


bool CompositeTriggerServiceTrigger::isXor()
{
   return ( ( (_triggerA->status() == _criterionA) && 
	    ((int)(*_serviceB) != _criterionB) )  ||
	    ( (_triggerA->status() != _criterionA) && 
	      ((int)(*_serviceB) == _criterionB) ) );
}

bool CompositeTriggerServiceTrigger::isSingle()
{
   return (_triggerA->status() == _criterionA);
}

void CompositeTriggerServiceTrigger::duplicate(std::unique_ptr<Trigger>& dup) const
{
   dup.reset(new CompositeTriggerServiceTrigger(*this));
}

CompositeTriggerServiceTrigger::~CompositeTriggerServiceTrigger()
{
}
