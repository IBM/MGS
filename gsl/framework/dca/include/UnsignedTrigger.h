// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef UNSIGNEDTRIGGER_H
#define UNSIGNEDTRIGGER_H
#include "Copyright.h"

#include "TriggerBase.h"

#include <string>
#include <list>
#include <vector>

class DataItem;
class Simulation;

class UnsignedTrigger : public TriggerBase
{

  public:
  UnsignedTrigger(Simulation& sim, std::vector<DataItem*> const& args);
  virtual bool status();
  virtual void duplicate(std::unique_ptr<Trigger>&& dup) const;
  virtual ~UnsignedTrigger();

  private:
  void setEvaluator(std::string op);
  bool isEqual();
  bool isGreaterThan();
  bool isLessThan();
  bool isNotEqual();
  bool isGreaterThanOrEqual();
  bool isLessThanOrEqual();
  bool isModulusZero();
  bool isModulusNonZero();

  Simulation& _sim;
  unsigned* _service;
  std::string _op;
  unsigned _criterion;
  bool (UnsignedTrigger::*_evaluator)();
};
#endif
