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

#include "EachAvgFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "Node.h"
#include "ConnectionContext.h"
#include "NodeSet.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "NumericDataItem.h"
#include "SyntaxErrorException.h"

EachAvgFunctor::EachAvgFunctor()
   : _nbrReps(0), _remainingProb(0), _nbrRepsDone(0), _count(0), 
     _combOffset(0), _phase(_REPETITIONS)
{
   _nodesIter = _nodesBegin = _nodes.begin();
   _nodesEnd = _nodes.end();
}

EachAvgFunctor::EachAvgFunctor(const EachAvgFunctor& csf)
   : _nodes(csf._nodes), _nbrReps(csf._nbrReps), 
     _remainingProb(csf._remainingProb), _nbrRepsDone(csf._nbrRepsDone), 
     _count(csf._count), _phase(csf._phase)
{
   _nodesIter = _nodesBegin = _nodes.begin();
   _nodesEnd = _nodes.end();
}

void EachAvgFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new EachAvgFunctor(*this));
}

EachAvgFunctor::~EachAvgFunctor()
{
}

void EachAvgFunctor::doInitialize(LensContext *c, 
				  const std::vector<DataItem*>& args)
{
   // Grab argument 
   if (args.size()!=1) {
      std::string mes = "EachAvg: invalid arguments!";
      mes += "\texpected: EachAvg(float avgNbrConnectionsPerNode)";
      throw SyntaxErrorException(mes);
   }
   NumericDataItem *avgDI = dynamic_cast<NumericDataItem*>(args[0]);
   if (avgDI==0) {
      throw SyntaxErrorException(
	 "EachAvg: argument 1 is not a NumericDataItem");
   }
   _avg=avgDI->getFloat();
   _nbrReps=int(_avg);
   _remainingProb=_avg-float(_nbrReps);
}

void EachAvgFunctor::doExecute(LensContext *c, 
			       const std::vector<DataItem*>& args, 
			       std::auto_ptr<DataItem>& rvalue)
{
   ConnectionContext* cc = c->connectionContext;
   NodeDescriptor** nodeSlot=0;
   NodeSet* nodeSet=0;

   switch(cc->current) {
      case ConnectionContext::_SOURCE:
         nodeSet = cc->sourceSet;
         nodeSlot = &cc->sourceNode;
         break;
      case ConnectionContext::_DEST:
         nodeSet = cc->destinationSet;
         nodeSlot = &cc->destinationNode;
         break;
      case ConnectionContext::_BOTH:
         throw SyntaxErrorException(
	    "EachAvgFunctor: invalid responsibility specification!");
   }
   cc->done = false;

   if (cc->restart) {
      _nodes.clear();
      nodeSet->getNodes(_nodes);
      _nodesIter = _nodesBegin = _nodes.begin();
      _nodesEnd = _nodes.end();
      _nbrRepsDone = 0;
      _count = 0;
//      _combOffset = Rangen.drandom32(0, 1.0 / _remainingProb);
      _combOffset = drandom(0, 1.0 / _remainingProb);
      _phase = _REPETITIONS;
      if (_remainingProb > 0) {    
         // shuffle
         for (int sz = _nodes.size() - 1; sz > 0; --sz) {
//            int draw = Rangen.irandom32(0, sz);
            int draw = irandom(0, sz);
            NodeDescriptor* n = _nodes[sz];
            _nodes[sz] = _nodes[draw];
            _nodes[draw] = n;
         }
      }
   }

   if (_phase == _REPETITIONS) {
      // still doing reps
      if (_nbrReps > _nbrRepsDone) {
	 ++_nbrRepsDone;
      } else {
	 // done with this set of reps for this node (
	 // since _nbrReps>_nbrRepsDone),
	 // or no reps necessary (since _nbrReps==_nbrRepsDone==0)

         // so increment the node iterator
         ++_nodesIter;           
	 
	 // and reset the reps counter to 1, since a node will be returned
         _nbrRepsDone = 1;         

	 // but if all reps for all nodes are done, or no reps are necessary
         if ( (_nodesIter == _nodesEnd) || (_nbrReps == 0) ) {     
	    // check if there's no need for probabilistic 
	    // sampling (i.e., you're done)
            if (_remainingProb == 0) {
	       _phase = _DONE;
            } else {
	       // if there is a need, set it up
               _phase = _PROBABILISTIC;
	    }
         }
      }
   }

   if (_phase == _PROBABILISTIC) { 
      // can only enter this if the *_nodesBegin has already been sample
      // probabilistically during last pass
      // increment nodes iterator along a comb sample w/o replacement
      _nodesIter = _nodesBegin + int(_combOffset + _count / _remainingProb);
      // check if you're at or past end of comb, if so you're done
      if (_nodesIter >= _nodesEnd) {
	 _phase = _DONE;
      }
      // increment the counter for number of probabilistic samples
      ++_count;                  
   }

   if (_phase != _DONE) {
      // set up node
      *nodeSlot = *_nodesIter;      
   } else { 
      // you're done
      nodeSlot = 0;
      cc->done = true;
   }
}
