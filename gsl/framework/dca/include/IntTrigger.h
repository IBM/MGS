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

#ifndef INTTRIGGER_H
#define INTTRIGGER_H
#include "Copyright.h"

#include "TriggerBase.h"

#include <string>
#include <list>
#include <vector>

class DataItem;
class Simulation;

class IntTrigger : public TriggerBase
{

   public:
      IntTrigger(Simulation& sim, std::vector<DataItem*> const & args);
      virtual bool status();
      virtual void duplicate(std::auto_ptr<Trigger>& dup) const;
      virtual ~IntTrigger();
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
      int* _service;
      std::string _op;
      int _criterion;
      bool (IntTrigger::*_evaluator) ();
};
#endif
