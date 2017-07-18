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

#include "UnsignedServiceTrigger.h"
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

UnsignedServiceTrigger::UnsignedServiceTrigger(Simulation& sim,
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
                 "UnsignedServiceTrigger!" << std::endl;
    exit(-1);
  }
  _description = descriptionDI->getString();
  ++iter;
  ServiceDataItem* ptrSvcDI = dynamic_cast<ServiceDataItem*>(*iter);
  if (ptrSvcDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to ServiceDataItem failed on "
                 "UnsignedServiceTrigger!" << std::endl;
    exit(-1);
  }
  ++iter;
  StringDataItem* opDI = dynamic_cast<StringDataItem*>(*iter);
  if (opDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to StringDataItem failed on "
                 "UnsignedServiceTrigger!" << std::endl;
    exit(-1);
  }
  _op = opDI->getString();
  ++iter;
  ServiceDataItem* ptrCriterionSvcDI = dynamic_cast<ServiceDataItem*>(*iter);
  if (ptrCriterionSvcDI == 0)
  {
    std::cerr << "Expect a ServiceDataItem on 4th argument of "
      "UnsignedServiceTrigger!" << std::endl;
    exit(-1);
  }
  ++iter;
  NumericDataItem* delayDI = dynamic_cast<NumericDataItem*>(*iter);
  if (delayDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to NumericDataItem failed on "
                 "UnsignedServiceTrigger!" << std::endl;
    exit(-1);
  }
  _delay = delayDI->getInt();
  ++iter;
  PhaseDataItem* phaseDI = dynamic_cast<PhaseDataItem*>(*iter);
  if (phaseDI == 0)
  {
    std::cerr << "Dynamic cast of DataItem to PhaseDataItem failed on "
                 "UnsignedServiceTrigger!" << std::endl;
    exit(-1);
  }
  _phase = phaseDI->getPhase()->getName();

  setEvaluator(_op);
  {
    GenericService<unsigned>* service;
    service = dynamic_cast<GenericService<unsigned>*>(ptrSvcDI->getService());
    assert(service != 0);
    // modification required to run in distributed computing environment -- Jizhu
    // Lu - Tuan 
    assert(service != NULL);
    _service = service->getData();

  }

  {
    GenericService<int>* service;
    service = 0;
    service = dynamic_cast<GenericService<int>*>(ptrCriterionSvcDI->getService());
    assert(service != 0);
    assert(service != NULL);
    _criterion = service->getData();
  }

  _sim.addTrigger(_phase, this);
}

bool UnsignedServiceTrigger::status()
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

void UnsignedServiceTrigger::setEvaluator(std::string op)
{
  if (_op == ">")
    _evaluator = &UnsignedServiceTrigger::isGreaterThan;
  else if (_op == "<")
    _evaluator = &UnsignedServiceTrigger::isLessThan;
  else if (_op == "==")
    _evaluator = &UnsignedServiceTrigger::isEqual;
  else if (_op == "!=")
    _evaluator = &UnsignedServiceTrigger::isNotEqual;
  else if ((_op == ">=") || (_op == "=>"))
    _evaluator = &UnsignedServiceTrigger::isGreaterThanOrEqual;
  else if ((_op == "<=") || (_op == "=<"))
    _evaluator = &UnsignedServiceTrigger::isLessThanOrEqual;
  else if (_op == "!%")
    _evaluator = &UnsignedServiceTrigger::isModulusZero;
  else if (_op == "%")
    _evaluator = &UnsignedServiceTrigger::isModulusNonZero;
  else
  {
    std::cerr << "Operation not supported on UnsignedServiceTrigger!" << std::endl;
    exit(-1);
  }
}

bool UnsignedServiceTrigger::isEqual() { return (*_service) == *_criterion; }

bool UnsignedServiceTrigger::isGreaterThan() { return (*_service) > *_criterion; }

bool UnsignedServiceTrigger::isLessThan() { return (*_service) < *_criterion; }

bool UnsignedServiceTrigger::isNotEqual() { return (*_service) != *_criterion; }

bool UnsignedServiceTrigger::isGreaterThanOrEqual()
{
  return (*_service) >= *_criterion;
}

bool UnsignedServiceTrigger::isLessThanOrEqual() { return (*_service) <= *_criterion; }

bool UnsignedServiceTrigger::isModulusZero() { return !((*_service) % *_criterion); }

bool UnsignedServiceTrigger::isModulusNonZero() { return (*_service) % *_criterion; }

void UnsignedServiceTrigger::duplicate(std::auto_ptr<Trigger>& dup) const
{
  dup.reset(new UnsignedServiceTrigger(*this));
}

UnsignedServiceTrigger::~UnsignedServiceTrigger() {}
