// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SEMAPHORETRIGGER_H
#define SEMAPHORETRIGGER_H
#include "Copyright.h"

#include "TriggerBase.h"

#include <string>
#include <vector>
#include <list>

class DataItem;
class Simulation;

class SemaphoreTrigger : public TriggerBase
{

   public:
      SemaphoreTrigger(Simulation& sim, std::vector<DataItem*> const & args);
      SemaphoreTrigger(Simulation& sim, Trigger* t1, Trigger* t2, 
		       std::string& op);
      SemaphoreTrigger(Simulation& sim, Trigger* t1);
      virtual bool status();
      virtual void duplicate(std::unique_ptr<Trigger>&& dup) const;
      virtual ~SemaphoreTrigger();
   private:
      void setEvaluator();
      bool isAnd();
      bool isOr();
      bool isXor();
      bool isSingle();

      Simulation& _sim;
      Trigger* _triggerA;
      Trigger* _triggerB;
      std::string _op;
      bool _criterionA;
      bool _criterionB;
      bool (SemaphoreTrigger::*_evaluator) ();
      bool _semaphore;
};
#endif
