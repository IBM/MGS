// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef COMPOSITETRIGGER_H
#define COMPOSITETRIGGER_H
#include "Copyright.h"

#include "TriggerBase.h"

#include <string>
#include <vector>
#include <list>

class DataItem;
class Simulation;

class CompositeTrigger : public TriggerBase
{

   public:
      CompositeTrigger(Simulation& sim, std::vector<DataItem*> const & args);
      CompositeTrigger(Simulation& sim, Trigger* t1, Trigger* t2, 
		       std::string& op);
      CompositeTrigger(Simulation& sim, Trigger* t1);
      virtual bool status();
      virtual void duplicate(std::unique_ptr<Trigger>&& dup) const;
      virtual ~CompositeTrigger();
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
      bool (CompositeTrigger::*_evaluator) ();
};
#endif
