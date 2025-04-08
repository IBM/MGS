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

#include "SemaphoreTrigger.h"
#include "Simulation.h"
#include "CustomStringDataItem.h"
#include "BoolDataItem.h"
#include "TriggerDataItem.h"
#include "PhaseDataItem.h"
#include "Phase.h"

// default criterionA & criterionB is true
SemaphoreTrigger::SemaphoreTrigger(Simulation& sim, 
				   std::vector<DataItem*> const & args)
   : _sim(sim)
{
//   if ((args.size() != 5) && (args.size() != 8)) {
//      std::cerr<<"Semaphore trigger accepts 5 or 8 arguments!"<<std::endl;
//      exit(-1);
//   }
   if ((args.size() != 5) ) {
      std::cerr<<"Semaphore trigger accepts 5 arguments!"<<std::endl;
      exit(-1);
   }
   std::vector<DataItem*>::const_iterator iter = args.begin();
   CustomStringDataItem* descriptionDI = dynamic_cast<CustomStringDataItem*>(*iter);
   if (descriptionDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to CustomStringDataItem failed on SemaphoreTrigger!"<<std::endl;
      exit(-1);
   }
   _description = descriptionDI->getString();
   ++iter;
   TriggerDataItem* triggerADI = dynamic_cast<TriggerDataItem*>(*iter);
   if (triggerADI == 0) {
      std::cerr<<"Dynamic cast of DataItem to TriggerDataItem failed on SemaphoreTrigger!"<<std::endl;
      exit(-1);
   }
   _triggerA = triggerADI->getTrigger();
   ++iter;
   NumericDataItem* criterionADI = dynamic_cast<NumericDataItem*>(*iter);
   if (criterionADI == 0) {
      std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on SemaphoreTrigger!"<<std::endl;
      exit(-1);
   }
   _criterionA = criterionADI->getBool();
   ++iter;

   if (args.size() == 8) {
//      CustomStringDataItem* opDI = dynamic_cast<CustomStringDataItem*>(*iter);
//      if (opDI == 0) {
//	 std::cerr<<"Dynamic cast of DataItem to CustomStringDataItem failed on SemaphoreTrigger!"<<std::endl;
//	 exit(-1);
//      }
//      _op = opDI->getString();
//      ++iter;
//      TriggerDataItem* triggerBDI = dynamic_cast<TriggerDataItem*>(*iter);
//      if (triggerBDI == 0) {
//	 std::cerr<<"Dynamic cast of DataItem to TriggerDataItem failed on SemaphoreTrigger!"<<std::endl;
//	 exit(-1);
//      }
//      _triggerB = triggerBDI->getTrigger();
//      ++iter;
//      NumericDataItem* criterionBDI = dynamic_cast<NumericDataItem*>(*iter);
//      if (criterionBDI == 0) {
//	 std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on SemaphoreTrigger!"<<std::endl;
//	 exit(-1);
//      }
//      _criterionB = criterionBDI->getBool();
//      ++iter;
   } else {
      _op = "single";
      _triggerB = 0;
      _criterionB = 1;
   }

   NumericDataItem* delayDI = dynamic_cast<NumericDataItem*>(*iter);
   if (delayDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on SemaphoreTrigger!"<<std::endl;
      exit(-1);
   }
   _delay = delayDI->getInt();
   ++iter;

   PhaseDataItem* phaseDI = dynamic_cast<PhaseDataItem*>(*iter);
   if (phaseDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to PhaseDataItem failed on SemaphoreTrigger!"<<std::endl;
      exit(-1);
   }
   _phase = phaseDI->getPhase()->getName();
   _semaphore = false;

   setEvaluator();
   _sim.addTrigger(_phase, this);
}

SemaphoreTrigger::SemaphoreTrigger(Simulation& sim, Trigger* t1, Trigger* t2, 
				   std::string& op)
   : _sim(sim), _triggerA(t1), _triggerB(t2), _op(op), _criterionA(1), 
     _criterionB(1), _semaphore(false)
{
   _description = "Auto Generated Semaphore Trigger";
   _delay= 0;
   setEvaluator();
   if ((t1 == 0) || (t2 == 0)) {
      std::cerr << "Warning: SemaphoreTrigger is constructed with a NULL pointer" << std::endl;
   }
   _phase = _sim.findLaterPhase(t1->getPhase(), t2->getPhase());
   _sim.addTrigger(_phase, this);
}

SemaphoreTrigger::SemaphoreTrigger(Simulation& sim, Trigger* t1)
   : _sim(sim), _triggerA(t1), _triggerB(0), _op("single"), 
     _criterionA(1), _criterionB(1), _semaphore(false)
{
   _description = "Auto Generated Semaphore Trigger";
   _delay= 0;
   setEvaluator();
   if (t1 == 0) {
      std::cerr << "Warning: SemaphoreTrigger is constructed with a NULL pointer" << std::endl;
   }
   _phase = t1->getPhase();
   _sim.addTrigger(_phase, this);
}

bool SemaphoreTrigger::status()
{
   bool ste = false;
   ste = (*this.*_evaluator)();
   
   if (not _semaphore)
   {
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
      if (_state)
	 _semaphore = true;
   }
   else
   {
      _state = false;
   }
   return _state;
}

void SemaphoreTrigger::setEvaluator()
{
   if ((_op == "&&") || (_op == "AND") || (_op == "and")) {
      _evaluator = &SemaphoreTrigger::isAnd;
   }
   else if ((_op == "||") || (_op == "OR") || (_op == "or")) {
      _evaluator = &SemaphoreTrigger::isOr;
   }
   else if ((_op == "XOR") || (_op == "xor")) {
      _evaluator = &SemaphoreTrigger::isXor;
   }
   else if ((_op == "SINGLE") || (_op == "single")) {
      _evaluator = &SemaphoreTrigger::isSingle;
   }
   else {
      std::cerr<<"Operation not supported on SemaphoreTrigger!"<<std::endl;
      exit(-1);
   }
}


bool SemaphoreTrigger::isAnd()
{
   return ( (_triggerA->status() == _criterionA) && 
	    (_triggerB->status() == _criterionB) );
}


bool SemaphoreTrigger::isOr()
{
   return ( (_triggerA->status() == _criterionA) || 
	    (_triggerB->status() == _criterionB) );
}


bool SemaphoreTrigger::isXor()
{
   return ( ( (_triggerA->status() == _criterionA) && 
	      (_triggerB->status() != _criterionB) ) ||
	    ( (_triggerA->status() != _criterionA) && 
	      (_triggerB->status() == _criterionB) ) );
}

bool SemaphoreTrigger::isSingle()
{
   return (_triggerA->status() == _criterionA);
}

void SemaphoreTrigger::duplicate(std::unique_ptr<Trigger>&& dup) const
{
   dup.reset(new SemaphoreTrigger(*this));
}

SemaphoreTrigger::~SemaphoreTrigger()
{
}
