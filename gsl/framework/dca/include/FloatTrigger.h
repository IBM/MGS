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

#ifndef FLOATTRIGGER_H
#define FLOATTRIGGER_H
#include "Copyright.h"

#include "TriggerBase.h"

#include <string>
#include <list>
#include <vector>

class DataItem;
class Simulation;

class FloatTrigger : public TriggerBase
{

   public:
      FloatTrigger(Simulation& sim, std::vector<DataItem*> const & args);
      virtual bool status();
      virtual void duplicate(std::unique_ptr<Trigger>& dup) const;
      virtual ~FloatTrigger();
   private:
      void setEvaluator(std::string op);
      bool isEqual();
      bool isGreaterThan();
      bool isLessThan();
      bool isNotEqual();
      bool isGreaterThanOrEqual();
      bool isLessThanOrEqual();

      Simulation& _sim;
      float* _service;
      std::string _op;
      float _criterion;
      bool (FloatTrigger::*_evaluator) ();
};
#endif
