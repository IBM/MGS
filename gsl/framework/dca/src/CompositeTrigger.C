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

#include "CompositeTrigger.h"
#include "Simulation.h"
#include "StringDataItem.h"
#include "BoolDataItem.h"
#include "TriggerDataItem.h"
#include "PhaseDataItem.h"
#include "Phase.h"

// default criterionA & criterionB is true
CompositeTrigger::CompositeTrigger(Simulation& sim,
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
      std::cerr<<"Dynamic cast of DataItem to StringDataItem failed on CompositeTrigger!"<<std::endl;
      exit(-1);
   }
   _description = descriptionDI->getString();
   ++iter;
   TriggerDataItem* triggerADI = dynamic_cast<TriggerDataItem*>(*iter);
   if (triggerADI == 0) {
      std::cerr<<"Dynamic cast of DataItem to TriggerDataItem failed on CompositeTrigger!"<<std::endl;
      exit(-1);
   }
   _triggerA = triggerADI->getTrigger();
   ++iter;
   NumericDataItem* criterionADI = dynamic_cast<NumericDataItem*>(*iter);
   if (criterionADI == 0) {
      std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on CompositeTrigger!"<<std::endl;
      exit(-1);
   }
   _criterionA = criterionADI->getBool();
   ++iter;

   if (args.size() == 8) {
      StringDataItem* opDI = dynamic_cast<StringDataItem*>(*iter);
      if (opDI == 0) {
	 std::cerr<<"Dynamic cast of DataItem to StringDataItem failed on CompositeTrigger!"<<std::endl;
	 exit(-1);
      }
      _op = opDI->getString();
      ++iter;
      TriggerDataItem* triggerBDI = dynamic_cast<TriggerDataItem*>(*iter);
      if (triggerBDI == 0) {
	 std::cerr<<"Dynamic cast of DataItem to TriggerDataItem failed on CompositeTrigger!"<<std::endl;
	 exit(-1);
      }
      _triggerB = triggerBDI->getTrigger();
      ++iter;
      NumericDataItem* criterionBDI = dynamic_cast<NumericDataItem*>(*iter);
      if (criterionBDI == 0) {
	 std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on CompositeTrigger!"<<std::endl;
	 exit(-1);
      }
      _criterionB = criterionBDI->getBool();
      ++iter;
   } else {
      _op = "single";
      _triggerB = 0;
      _criterionB = 1;
   }

   NumericDataItem* delayDI = dynamic_cast<NumericDataItem*>(*iter);
   if (delayDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to NumericDataItem failed on CompositeTrigger!"<<std::endl;
      exit(-1);
   }
   _delay = delayDI->getInt();
   ++iter;

   PhaseDataItem* phaseDI = dynamic_cast<PhaseDataItem*>(*iter);
   if (phaseDI == 0) {
      std::cerr<<"Dynamic cast of DataItem to PhaseDataItem failed on CompositeTrigger!"<<std::endl;
      exit(-1);
   }
   _phase = phaseDI->getPhase()->getName();

   setEvaluator();
   _sim.addTrigger(_phase, this);
}

CompositeTrigger::CompositeTrigger(Simulation& sim, Trigger* t1, Trigger* t2,
				   std::string& op)
   : _sim(sim), _triggerA(t1), _triggerB(t2), _op(op), _criterionA(1),
     _criterionB(1)
{
   _description = "Auto Generated Composite Trigger";
   _delay= 0;
   setEvaluator();
   if ((t1 == 0) || (t2 == 0)) {
      std::cerr << "Warning: CompositeTrigger is constructed with a NULL pointer" << std::endl;
   }
   _phase = _sim.findLaterPhase(t1->getPhase(), t2->getPhase());
   _sim.addTrigger(_phase, this);
}

CompositeTrigger::CompositeTrigger(Simulation& sim, Trigger* t1)
   : _sim(sim), _triggerA(t1), _triggerB(0), _op("single"),
     _criterionA(1), _criterionB(1)
{
   _description = "Auto Generated Composite Trigger";
   _delay= 0;
   setEvaluator();
   if (t1 == 0) {
      std::cerr << "Warning: CompositeTrigger is constructed with a NULL pointer" << std::endl;
   }
   _phase = t1->getPhase();
   _sim.addTrigger(_phase, this);
}

bool CompositeTrigger::status()
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

void CompositeTrigger::setEvaluator()
{
   if ((_op == "&&") || (_op == "AND") || (_op == "and")) {
      _evaluator = &CompositeTrigger::isAnd;
   }
   else if ((_op == "||") || (_op == "OR") || (_op == "or")) {
      _evaluator = &CompositeTrigger::isOr;
   }
   else if ((_op == "XOR") || (_op == "xor")) {
      _evaluator = &CompositeTrigger::isXor;
   }
   else if ((_op == "SINGLE") || (_op == "single")) {
      _evaluator = &CompositeTrigger::isSingle;
   }
   else {
      std::cerr<<"Operation not supported on CompositeTrigger!"<<std::endl;
      exit(-1);
   }
}


bool CompositeTrigger::isAnd()
{
   return ( (_triggerA->status() == _criterionA) &&
	    (_triggerB->status() == _criterionB) );
}


bool CompositeTrigger::isOr()
{
   return ( (_triggerA->status() == _criterionA) ||
	    (_triggerB->status() == _criterionB) );
}


bool CompositeTrigger::isXor()
{
   return ( ( (_triggerA->status() == _criterionA) &&
	      (_triggerB->status() != _criterionB) ) ||
	    ( (_triggerA->status() != _criterionA) &&
	      (_triggerB->status() == _criterionB) ) );
}

bool CompositeTrigger::isSingle()
{
   return (_triggerA->status() == _criterionA);
}

void CompositeTrigger::duplicate(std::auto_ptr<Trigger>& dup) const
{
   dup.reset(new CompositeTrigger(*this));
}

CompositeTrigger::~CompositeTrigger()
{
}
