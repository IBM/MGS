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

#include "UnsignedTrigger.h"
#include "DataItem.h"
#include "UnsignedIntArrayDataItem.h"
#include "NumericDataItem.h"
#include "Simulation.h"
#include "Service.h"
#include "StringDataItem.h"
#include "BoolDataItem.h"
#include "TriggerDataItem.h"
#include "ServiceDataItem.h"
#include "PhaseDataItem.h"
#include "Phase.h"
#include "GenericService.h"

#include <cassert>

UnsignedTrigger::UnsignedTrigger(Simulation& sim,
                                 std::vector<DataItem*> const& args)
    : TriggerBase(), _sim(sim)
{
  if (args.size() != 6)
  {
    std::cerr << "Unsigned trigger accepts 6 arguments!" << std::endl;
    exit(-1);
  }
  std::vector<DataItem*>::const_iterator iter = args.begin();
  StringDataItem* descriptionDI = dynamic_cast<StringDataItem*>(*iter);
  if (descriptionDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to StringDataItem failed on "
                 "UnsignedTrigger!" << std::endl;
    exit(-1);
  }
  _description = descriptionDI->getString();
  ++iter;
  ServiceDataItem* ptrSvcDI = dynamic_cast<ServiceDataItem*>(*iter);
  if (ptrSvcDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to ServiceDataItem failed on "
                 "UnsignedTrigger!" << std::endl;
    exit(-1);
  }
  ++iter;
  StringDataItem* opDI = dynamic_cast<StringDataItem*>(*iter);
  if (opDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to StringDataItem failed on "
                 "UnsignedTrigger!" << std::endl;
    exit(-1);
  }
  _op = opDI->getString();
  ++iter;
  NumericDataItem* criterionDI = dynamic_cast<NumericDataItem*>(*iter);
  if (criterionDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to NumericDataItem failed on "
                 "UnsignedTrigger!" << std::endl;
    exit(-1);
	}
	_criterion = criterionDI->getUnsignedInt();
	//MODIFIED: either a NumericDataItem or a ServiceDataItem
	/*
  if (criterionDI == 0)
	{
		ServiceDataItem* ptrSvcDI = dynamic_cast<ServiceDataItem*>(*iter);
		if (ptrSvcDI == 0)
		{
			std::cerr << "Expect a NumericDataItem or ServiceDataItem on 4th argument of "
				"UnsignedTrigger!" << std::endl;
			exit(-1);
		}
		else{
			std::cerr << "WARNING: You are passing a ServiceDataItem " <<
				"which will be converted to NumericDataItem in "
				"UnsignedTrigger!" << std::endl;
			GenericService<int>* service;
			service = dynamic_cast<GenericService<int>*>(ptrSvcDI->getService());
			assert(service != 0);
			assert(service != NULL);
			_criterion = *(service->getData());
		}
	}
	else{
		_criterion = criterionDI->getUnsignedInt();

	}*/
  ++iter;
  NumericDataItem* delayDI = dynamic_cast<NumericDataItem*>(*iter);
  if (delayDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to NumericDataItem failed on "
                 "UnsignedTrigger!" << std::endl;
    exit(-1);
  }
  _delay = delayDI->getInt();
  ++iter;
  PhaseDataItem* phaseDI = dynamic_cast<PhaseDataItem*>(*iter);
  if (phaseDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to PhaseDataItem failed on "
                 "UnsignedTrigger!" << std::endl;
    exit(-1);
  }
  _phase = phaseDI->getPhase()->getName();

  setEvaluator(_op);
  GenericService<unsigned>* service;
  service = dynamic_cast<GenericService<unsigned>*>(ptrSvcDI->getService());
  assert(service != 0);
  // modification required to run in distributed computing environment -- Jizhu
  // Lu on 03/29/2006
  assert(service != NULL);
  _service = service->getData();
  _sim.addTrigger(_phase, this);
}

bool UnsignedTrigger::status()
{
  bool ste = false;

  ste = (*this.*_evaluator)();

  if (_delay > 0)
  {
    if (_stateHistory.size() <= _delay)
    {
      _stateHistory.push_front(ste);
      _state = false;
    }
    else
    {
      _stateHistory.pop_back();
      _stateHistory.push_front(ste);
      _state = _stateHistory.back();
    }
  }
  else
  {
    _state = ste;
  }
  return _state;
}

void UnsignedTrigger::setEvaluator(std::string op)
{
  if (_op == ">")
    _evaluator = &UnsignedTrigger::isGreaterThan;
  else if (_op == "<")
    _evaluator = &UnsignedTrigger::isLessThan;
  else if (_op == "==")
    _evaluator = &UnsignedTrigger::isEqual;
  else if (_op == "!=")
    _evaluator = &UnsignedTrigger::isNotEqual;
  else if ((_op == ">=") || (_op == "=>"))
    _evaluator = &UnsignedTrigger::isGreaterThanOrEqual;
  else if ((_op == "<=") || (_op == "=<"))
    _evaluator = &UnsignedTrigger::isLessThanOrEqual;
  else if (_op == "!%")
    _evaluator = &UnsignedTrigger::isModulusZero;
  else if (_op == "%")
    _evaluator = &UnsignedTrigger::isModulusNonZero;
  else
  {
    std::cerr << "Operation not supported on UnsignedTrigger!" << std::endl;
    exit(-1);
  }
}

bool UnsignedTrigger::isEqual() { return (*_service) == _criterion; }

bool UnsignedTrigger::isGreaterThan() { return (*_service) > _criterion; }

bool UnsignedTrigger::isLessThan() { return (*_service) < _criterion; }

bool UnsignedTrigger::isNotEqual() { return (*_service) != _criterion; }

bool UnsignedTrigger::isGreaterThanOrEqual()
{
  return (*_service) >= _criterion;
}

bool UnsignedTrigger::isLessThanOrEqual() { return (*_service) <= _criterion; }

bool UnsignedTrigger::isModulusZero() { return !((*_service) % _criterion); }

bool UnsignedTrigger::isModulusNonZero() { return (*_service) % _criterion; }

void UnsignedTrigger::duplicate(std::auto_ptr<Trigger>& dup) const
{
  dup.reset(new UnsignedTrigger(*this));
}

UnsignedTrigger::~UnsignedTrigger() {}
