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
      virtual void duplicate(std::auto_ptr<Trigger>& dup) const;
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
