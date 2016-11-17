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
  virtual void duplicate(std::auto_ptr<Functor>& fap) const;
  virtual ~EachAvgFunctor();

  protected:
  virtual void doInitialize(LensContext* c, const std::vector<DataItem*>& args);
  virtual void doExecute(LensContext* c, const std::vector<DataItem*>& args,
                         std::auto_ptr<DataItem>& rvalue);

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
