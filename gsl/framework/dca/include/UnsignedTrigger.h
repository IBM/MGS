// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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
  virtual void duplicate(std::auto_ptr<Trigger>& dup) const;
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
