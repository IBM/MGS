// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BOOLTRIGGER_H
#define BOOLTRIGGER_H
#include "Copyright.h"

#include "TriggerBase.h"

#include <string>
#include <list>
#include <vector>

class DataItem;
class Simulation;

class BoolTrigger : public TriggerBase
{

   public:
      BoolTrigger(Simulation& sim, std::vector<DataItem*> const & args);
      virtual bool status();
      virtual void duplicate(std::unique_ptr<Trigger>&& dup) const;
      virtual ~BoolTrigger();
   private:
      void setEvaluator(std::string op);
      bool isEqual();
      bool isNotEqual();

      Simulation& _sim;
      bool* _service;
      std::string _op;
      bool _criterion;
      bool (BoolTrigger::*_evaluator) ();
};
#endif
