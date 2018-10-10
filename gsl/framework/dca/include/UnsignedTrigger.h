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
  virtual void duplicate(std::unique_ptr<Trigger>& dup) const;
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
