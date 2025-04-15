// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "RadialHistoSamplerFunctor.h"
#include "LensContext.h"
#include "ConnectionContext.h"
#include "DataItem.h"
#include "IntArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "NumericDataItem.h"
#include "FunctorType.h"
#include "GridLayerDescriptor.h"
#include "SurfaceOdometer.h"
#include "VolumeOdometer.h"
#include "Grid.h"
#include "Simulation.h"
#include "Node.h"
#include "NodeSet.h"
#include "NodeAccessor.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "rndm.h"
#include "SyntaxErrorException.h"
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

RadialHistoSamplerFunctor::RadialHistoSamplerFunctor()
   : _nbrSamples(0), _radialScale(0), _borderTolerance(0), 
     _sampleSet(), _list(),_responsibility(ConnectionContext::_BOTH),
     _currentSample(0), _refNode(0), _currentCount(0), _combCount(0), 
     _combSpacing(0), _combSize(0), _intervalIndex(0), _intervalOffset(0), 
     _sampleSpace(), _probSum(0)
{
}

void RadialHistoSamplerFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new RadialHistoSamplerFunctor(*this));
}


RadialHistoSamplerFunctor::~RadialHistoSamplerFunctor()
{
}


void RadialHistoSamplerFunctor::doInitialize(
   LensContext *c, const std::vector<DataItem*>& args)
{
   /*
   Grab the int list
   Copy and store
   Set up current count lists to initial condition
   */

   // prototype SamplingFctr2 RadialHistoSampler(list<int>,int direction);
   int nbrArgs=args.size();
   if (nbrArgs!=3 && nbrArgs!=4) {
      std::ostringstream msg;
      msg << "RadialHistoSampler: invalid number of arguments" 
	  << std::endl
	  << "\texpected: RadialHistoSampler(int nbrConnections, float radialScale, list<float> function), or" 
	  << "\t          RadialHistoSampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)."
	  << std::endl;
      throw SyntaxErrorException(msg.str());
   }

   NumericDataItem *nbrSamplesDI = dynamic_cast<NumericDataItem*>(args[0]);
   if (nbrSamplesDI==0) {
      std::ostringstream msg;
      msg << "RadialHistoSampler: argument 1 is not a NumericDataItem" 
	  << std::endl
	  << "\texpected: RadialHistoSampler(int nbrConnections, float radialScale, list<float> function) or" 
	  << "\t          RadialHistoSampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)."
	  << std::endl;
      throw SyntaxErrorException(msg.str());
   }
   _nbrSamples=unsigned(nbrSamplesDI->getInt());
   NumericDataItem *radialScaleDI = dynamic_cast<NumericDataItem*>(args[1]);
   if (radialScaleDI==0) {
      std::ostringstream msg;
      msg << "RadialHistoSampler: argument 2 is not a NumericDataItem" 
	  << std::endl
	  << "\texpected: RadialHistoSampler(int nbrConnections, float radialScale, list<float> function) or" 
	  << "\t          RadialHistoSampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)." 
	  << std::endl;
      throw SyntaxErrorException(msg.str());
   }
   _radialScale=radialScaleDI->getFloat();
   FloatArrayDataItem *fa_di = dynamic_cast<FloatArrayDataItem*>(args[2]);
   if (fa_di==0) {
      std::ostringstream msg;
      msg << "RadialHistoSamplerFunctor: argument 3 is not list<float>" 
	  << std::endl
	  << "\texpected: RadialHistoSampler(int nbrConnections, float radialScale, list<float> function) or" 
	  << "\t          RadialHistoSampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)."
	  << std::endl;
      throw SyntaxErrorException(msg.str());
   }
   _list = *fa_di->getFloatVector();
   if (_list.size()<2) {
      throw SyntaxErrorException(
	 "RadialHistoSamplerFunctor: list must have at least two elements");
   }
   if (nbrArgs==4) {
     NumericDataItem *borderToleranceDI = 
	dynamic_cast<NumericDataItem*>(args[3]);
     if (borderToleranceDI==0) {
       std::ostringstream msg;
       msg << "RadialHistoSampler: argument 4 is not a NumericDataItem" 
	   << std::endl
	   << "\texpected: RadialHistoSampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)."
	   << std::endl;
       throw SyntaxErrorException(msg.str());
     }
     _borderTolerance=unsigned(borderToleranceDI->getInt());
   }

   std::vector<float>::iterator i1, i2, begin = _list.begin(), 
      end = _list.end();
   float prevCummulative=0;
   i1 = begin;
   i2 = i1+1;
   float xIncrement = 1.0/(_list.size()-1.0);
   int n = 0;
   for (;i2!=end; ++i1, ++i2) {
      CummulativeProbabilitySegment cps;
      float x1 = n * xIncrement;
      float x2 = (n + 1) * xIncrement;
      cps.prevCummulative = prevCummulative;
      cps.slope = (*i2 - *i1) / xIncrement;
      cps.offset = *i1 - cps.slope * x1;
      prevCummulative += 
	 0.5 * cps.slope * (x2*x2 - x1*x1) + cps.offset * (x2-x1);
      ++n;
      _cummProbSeg.push_back(cps);
   }

}


float RadialHistoSamplerFunctor::getCummulativeProbability(
   float distance, float scale, 
   std::vector<CummulativeProbabilitySegment> &segs)
{
   float x = distance/scale;
   if (x>1) x=1;                 // should never get here

   int index = int(x*segs.size());
   if (x==1) {
      --index;            // go to last segment if at very end
   }
   float x1 = float(index) /float(segs.size());
   return (segs[index].prevCummulative + 0.5 * 
	   segs[index].slope * (x*x - x1*x1) + segs[index].offset*(x-x1) );
}


void RadialHistoSamplerFunctor::doExecute(
   LensContext *c, const std::vector<DataItem*>& args, 
   std::unique_ptr<DataItem>& rvalue)
{
   ConnectionContext *cc = c->connectionContext;
   ConnectionContext::Responsibility resp = cc->current;
   bool refNodeDifferent = false;
   NodeSet* source=0;
   NodeDescriptor** slot=0;

   switch(resp) {
      case ConnectionContext::_SOURCE:
         source = cc->sourceSet;
         slot = &cc->sourceNode;
         if(_refNode != cc->sourceRefNode) {
            _refNode = cc->sourceRefNode;
            refNodeDifferent = true;
         }
         break;
      case ConnectionContext::_DEST:
         source = cc->destinationSet;
         slot = &cc->destinationNode;
         if(_refNode != cc->destinationRefNode) {
            _refNode = cc->destinationRefNode;
            refNodeDifferent = true;
         }
         break;
      case ConnectionContext::_BOTH:
         throw SyntaxErrorException(
	    "RadialHistoSamplerFunctor: invalid responsibility specification");
   }

   if (cc->restart) {
      _sampleSpace.clear();
      std::vector<int> coords, refcoords, begincoords, endcoords, 
	 mincoords, maxcoords, gridSize;
      int radius = int(ceil(_radialScale));
      _refNode->getNodeCoords(refcoords);
      mincoords = source->getBeginCoords();
      maxcoords = source->getEndCoords();
      int min, max, minTolerated, maxTolerated, absMax;
      gridSize = source->getGrid()->getSize();
      for(unsigned i=0;i<refcoords.size();++i) {
         min = refcoords[i] - radius;
         max = refcoords[i] + radius;
 	 minTolerated = mincoords[i]-_borderTolerance;
	 maxTolerated = maxcoords[i]+_borderTolerance;
	 absMax = gridSize[i]-1;
	 
	 min = (min<0)? 0:min;
	 max = (max>absMax)? absMax:max;

         begincoords.push_back((min<minTolerated)? minTolerated:min);
         endcoords.push_back((max>maxTolerated)? maxTolerated:max);
      }

      //gather nodes and probabilities
      float distance, dd;
      std::vector<Sample> tmpSampleSpace;
      VolumeOdometer odmtr(begincoords, endcoords);
      for (coords = odmtr.look(); !odmtr.isRolledOver(); 
	   coords = odmtr.next() ) {
         distance = 0;
         for(unsigned i=0;i<coords.size();++i) {
            dd = refcoords[i] - coords[i];
            distance += dd*dd;
         }
         distance=sqrt(distance);

         if (distance <=_radialScale) {
            Sample s;
            std::vector<NodeDescriptor*> nodes;
            NodeSet ns(*source);
            ns.setCoords(coords, coords);
            ns.getNodes(nodes);
            unsigned density =  nodes.size();
            for(unsigned i=0;i<density;++i) {
               s.node = nodes[i];
               s.relProb = 0;
               s.distance = distance;
               tmpSampleSpace.push_back(s);
            }
         }
      }
      if (tmpSampleSpace.size() ==0) {
         throw SyntaxErrorException(
	    "RadialHistoSample: There are no nodes to sample from");
      }

      // sort tmpSampleSpace
      sort(tmpSampleSpace.begin(),tmpSampleSpace.end());

      // assign probabilities
      float prob;
      float prevDistance = 0;
      float prevCummulative = 0;
      float nextDistance = 0;
      float nextCummulative = 0;
      int tbottom = 0, ttop = 0;
      int end = tmpSampleSpace.size();
      float currDistance = tmpSampleSpace[0].distance;

      while (ttop!=end) {
         // skip to the last sample at the current distance
         while(
	    ttop!=end && tmpSampleSpace[ttop].distance ==currDistance) ++ttop;

         // compute top distance
         if (ttop !=end) {
            currDistance = tmpSampleSpace[ttop].distance;
            nextDistance = (currDistance + prevDistance)/2;
         }
         else
            nextDistance = _radialScale;

         // compute the probability
         nextCummulative = getCummulativeProbability(
	    nextDistance, _radialScale, _cummProbSeg);
         prob = (nextCummulative - prevCummulative)/(ttop - tbottom);

         // assign the computed probability
         for (int i =tbottom;i<ttop; ++i)
            tmpSampleSpace[i].relProb = prob;

         prevDistance = nextDistance;
         prevCummulative = nextCummulative;
         tbottom = ttop;
      }

      // load non-zero probability samples into _sampleSpace
      _sampleSpace.clear();
      for (int i=0;i<end;++i) {
         if (tmpSampleSpace[i].relProb >0) 
	    _sampleSpace.push_back(tmpSampleSpace[i]);
      }
      if (_sampleSpace.size() ==0) {
         throw SyntaxErrorException(
	    "RadialHistoSample: There are no nodes with positive probability to sample from");
      }

      // shuffle
      Sample tmp;
      unsigned pick;
      unsigned top = _sampleSpace.size() -1;
      while (top>0) {
         pick = irandom(0,top,c->sim->getSharedFunctorRandomSeedGenerator());
         tmp = _sampleSpace[top];
         _sampleSpace[top] = _sampleSpace[pick];
         _sampleSpace[pick] = tmp;
         --top;
      };

      // Set up sampling comb
      _probSum=0;
      for (unsigned u = 0;u<_sampleSpace.size();++u) {
         _probSum += _sampleSpace[u].relProb;
      }
      _combSpacing = _probSum/(_nbrSamples);
      _combSize = _nbrSamples;
      _combCount = 0;

      // set remaining variables
      _intervalOffset = _sampleSpace[0].relProb;
      _intervalIndex = 0;
      _responsibility = resp;
      _currentCount = 0;
      _currentSample = cc->currentSample;
   }

   if (_currentSample != cc->currentSample || _combCount>=_combSize) {
      _currentSample = cc->currentSample;
      ++_currentCount;
      if (_combCount >= _combSize &&_currentCount<_nbrSamples) {
         // reshuffle, recompute comb spacing
         Sample tmp;
         unsigned pick;
         unsigned top = _sampleSpace.size() -1;
         while (top>0) {
            pick = irandom(0,top,c->sim->getSharedFunctorRandomSeedGenerator());
            tmp = _sampleSpace[top];
            _sampleSpace[top] = _sampleSpace[pick];
            _sampleSpace[pick] = tmp;
            --top;
         };
         _intervalOffset = _sampleSpace[0].relProb;
         _intervalIndex = 0;

         // Set up sampling comb
         _combSpacing = _probSum/(_nbrSamples-_currentCount);
         _combSize = _nbrSamples-_currentCount;
         _combCount =0;

      }

   }
   if (_currentCount >= _nbrSamples) {
      *slot = 0;
      cc->done = true;
      return;
   }

   // sample, set, and return

   // draw sample with comb
   double combOffset = _combCount * _combSpacing;
   while(_intervalOffset < combOffset) {
      _intervalOffset+=_sampleSpace[++_intervalIndex].relProb;
   };
   *slot = _sampleSpace[_intervalIndex].node;
   ++_combCount;
   cc->done = false;

}


/* Comments for doExecute above
grab currentSample from ConnectionContext
grab responsibility
if (Reference node or responsibility is different){ // begin state for a particular reference node
   save new reference point
   Grab a list of nodes for each ring
   set current list number
   set current counts

   randomly select with replacement from current list
   set node of responsibility on the connection context
return
}
if (currentSample different){
increment current count of current list
if (at end of current list) set current list to next list
if (at end of last list) set node of responsibility to 0 and return
}
randomly select with replacement from current list and set node of responsibility on the connection context
return

*/
