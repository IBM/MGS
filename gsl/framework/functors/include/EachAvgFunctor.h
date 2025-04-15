// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _EACHAVGFUNCTOR_H_
#define _EACHAVGFUNCTOR_H_
#include "Copyright.h"

#include "SampFctr1Functor.h"
#include "rndm.h"
#include <memory>
#include <list>
#include <vector>

class DataItem;
class LensContext;
class Functor;
class NodeDescriptor;
class NodeSet;

class EachAvgFunctor : public SampFctr1Functor
{
  public:
  enum SamplingPhase
  {
    _REPETITIONS,
    _PROBABILISTIC,
    _DONE
  };
  EachAvgFunctor();
  EachAvgFunctor(const EachAvgFunctor&);
  virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
  virtual ~EachAvgFunctor();

  protected:
  virtual void doInitialize(LensContext* c, const std::vector<DataItem*>& args);
  virtual void doExecute(LensContext* c, const std::vector<DataItem*>& args,
                         std::unique_ptr<DataItem>& rvalue);

  private:
  std::vector<NodeDescriptor*> _nodes;
  std::vector<NodeDescriptor*>::iterator _nodesIter, _nodesBegin, _nodesEnd;
  //float _avg;
  int _nbrReps;
  float _remainingProb;
  int _nbrRepsDone;
  int _count;
  float _combOffset;
  SamplingPhase _phase;
};
#endif
