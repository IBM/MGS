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
      virtual void duplicate(std::auto_ptr<Trigger>& dup) const;
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
