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
      virtual void duplicate(std::unique_ptr<Trigger>& dup) const;
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
