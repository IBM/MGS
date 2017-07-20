// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef DOUBLETRIGGER_H
#define DOUBLETRIGGER_H
#include "Copyright.h"

#include "TriggerBase.h"

#include <string>
#include <list>
#include <vector>

class DataItem;
class Simulation;

class DoubleTrigger : public TriggerBase
{

   public:
      DoubleTrigger(Simulation& sim, std::vector<DataItem*> const & args);
      virtual bool status();
      virtual void duplicate(std::auto_ptr<Trigger>& dup) const;
      virtual ~DoubleTrigger();
   private:
      void setEvaluator(std::string op);
      bool isEqual();
      bool isGreaterThan();
      bool isLessThan();
      bool isNotEqual();
      bool isGreaterThanOrEqual();
      bool isLessThanOrEqual();

      Simulation& _sim;
      double* _service;
      std::string _op;
      double _criterion;
      bool (DoubleTrigger::*_evaluator) ();
};
#endif
