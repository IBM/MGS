// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef COMPOSITETRIGGERSERVICETRIGGER_H
#define COMPOSITETRIGGERSERVICETRIGGER_H
#include "Copyright.h"

#include "TriggerBase.h"
#include "Service.h"

#include <string>
#include <vector>
#include <list>

class DataItem;
class Simulation;

class CompositeTriggerServiceTrigger : public TriggerBase
{

   public:
      CompositeTriggerServiceTrigger(Simulation& sim, std::vector<DataItem*> const & args);
      CompositeTriggerServiceTrigger(Simulation& sim, Trigger* t1, unsigned* service, 
		       std::string& op);
      CompositeTriggerServiceTrigger(Simulation& sim, Trigger* t1);
      virtual bool status();
      virtual void duplicate(std::unique_ptr<Trigger>&& dup) const;
      virtual ~CompositeTriggerServiceTrigger();
   private:
      void setEvaluator();
      bool isAnd();
      bool isOr();
      bool isXor();
      bool isSingle();

      Simulation& _sim;
      Trigger* _triggerA;
      //Trigger* _triggerB;
      unsigned* _serviceB;
      std::string _op;
      bool _criterionA;
      bool _criterionB;
      bool (CompositeTriggerServiceTrigger::*_evaluator) ();
};
#endif
