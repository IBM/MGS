// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TRIGGERBASE_H
#define TRIGGERBASE_H
#include "Copyright.h"

#include "Trigger.h"
#include "Triggerable.h"
#include "TriggerableCaller.h"
#include "WorkUnit.h"

#include <list>
#include <string>
//#include <iostream>
#include <deque>

class TriggerBase : public Trigger {
  public:
  TriggerBase();
  TriggerBase(const TriggerBase& rv);
  TriggerBase& operator=(const TriggerBase& rv);

  virtual void conditionalFire();
  virtual void fireSerial();
  virtual void setDelay(unsigned delay) { _delay = delay; }
  virtual unsigned getDelay() { return _delay; }
  virtual void addSerialTriggerableCaller(
      std::unique_ptr<TriggerableCaller>& triggerableCaller) {
    _serialTriggerableCallers.push_back(triggerableCaller.release());
  }
  virtual void addParallelTriggerableCaller(
      std::unique_ptr<TriggerableCaller>& triggerableCaller) {
    _parallelTriggerableCallers.push_back(triggerableCaller.release());
    partitionWorkUnits();
  }
  virtual std::string getDescription() { return _description; }
  virtual void setNumOfThreads(int numOfThreads) {
    _numOfThreads = numOfThreads;
    partitionWorkUnits();
  }
  virtual std::deque<WorkUnit*>& getWorkUnits() { return _workUnits; }
  virtual std::string getPhase() const { return _phase; }
  virtual ~TriggerBase();

  protected:
  void copyOwnedHeap(const TriggerBase& rv);
  void destructOwnedHeap();
  inline void destructWorkUnits();

  unsigned _delay;
  bool _state;
  std::list<bool> _stateHistory;
  std::string _description;
  std::vector<TriggerableCaller*> _serialTriggerableCallers;
  std::vector<TriggerableCaller*> _parallelTriggerableCallers;
  int _numOfThreads;
  std::string _phase;

  private:
  void partitionWorkUnits();

  std::deque<WorkUnit*> _workUnits;
};

#endif
