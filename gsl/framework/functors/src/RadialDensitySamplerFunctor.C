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

#include "RadialDensitySamplerFunctor.h"
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
#include <stdlib.h>
#include <sstream>
#include <math.h>

RadialDensitySamplerFunctor::RadialDensitySamplerFunctor()
   : _nbrSamples(0), _radialScale(0), _borderTolerance(0), _sampleSet(), 
     _list(),_responsibility(ConnectionContext::_BOTH), _currentSample(0), 
     _refNode(0), _currentCount(0), _combCount(0), _combSpacing(0), 
     _combSize(0), _intervalIndex(0), _intervalOffset(0), _sampleSpace(), 
     _probSum(0)
{
}

void RadialDensitySamplerFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new RadialDensitySamplerFunctor(*this));
}


RadialDensitySamplerFunctor::~RadialDensitySamplerFunctor()
{
}


void RadialDensitySamplerFunctor::doInitialize(
   LensContext *c, const std::vector<DataItem*>& args)
{
   /*
   Grab the int list
   Copy and store
   Set up current count lists to initial condition
   */
  
   // prototype SamplingFctr2 RadialDensitySampler(list<int>,int direction);

   int nbrArgs=args.size();
   if (nbrArgs!=3 && nbrArgs!=4) {
      std::ostringstream msg;
      msg << "RadialDensitySampler: invalid number of arguments" << std::endl
	  << "\texpected: RadialDensitySampler(int nbrConnections, float radialScale, list<float> function), or" 
	  << "\t          RadialDensitySampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)."
	  << std::endl;
      throw SyntaxErrorException(msg.str());
   }

   NumericDataItem *nbrSamplesDI = dynamic_cast<NumericDataItem*>(args[0]);
   if (nbrSamplesDI==0) {
      std::ostringstream msg;
      msg << "RadialDensitySampler: argument 1 is not a NumericDataItem" 
	  << std::endl
	 << "\texpected: RadialDensitySampler(int nbrConnections, float radialScale, list<float> function) or" 
	 << "\t          RadialDensitySampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)."<<std::endl;
      throw SyntaxErrorException(msg.str());
   }
   _nbrSamples=unsigned(nbrSamplesDI->getInt());
   NumericDataItem *radialScaleDI = dynamic_cast<NumericDataItem*>(args[1]);
   if (radialScaleDI==0) {
      std::ostringstream msg;
      msg << "RadialDensitySampler: argument 2 is not a NumericDataItem" << std::endl
	 << "\texpected: RadialDensitySampler(int nbrConnections, float radialScale, list<float> function) or" 
	 << "\t          RadialDensitySampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)."<<std::endl;
      throw SyntaxErrorException(msg.str());
   }
   _radialScale=radialScaleDI->getFloat();
   FloatArrayDataItem *fa_di = dynamic_cast<FloatArrayDataItem*>(args[2]);
   if (fa_di==0) {
      std::ostringstream msg;
      msg << "RadialDensitySamplerFunctor: argument 3 is not list<float>" << std::endl
	 << "\texpected: RadialDensitySampler(int nbrConnections, float radialScale, list<float> function) or" 
	 << "\t          RadialDensitySampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)."<<std::endl;
      throw SyntaxErrorException(msg.str());
   }
   _list = *fa_di->getFloatVector();
   if (nbrArgs==4) {
     NumericDataItem *borderToleranceDI = dynamic_cast<NumericDataItem*>(args[3]);
     if (borderToleranceDI==0) {
       std::ostringstream msg;
       msg << "RadialDensitySampler: argument 4 is not a NumericDataItem" << std::endl
	 << "\texpected: RadialDensitySampler(int nbrConnections, float radialScale, list<float> function, int borderTolerance)."<<std::endl;
       throw SyntaxErrorException(msg.str());
     }
     _borderTolerance=unsigned(borderToleranceDI->getInt());
   }
}


float RadialDensitySamplerFunctor::getRelativeProbability(
   float distance, float scale, std::vector<float>& histo)
{
   float retval =0;
   float portion = distance/scale;
   if (portion <1) {
      float pointer = portion  * (histo.size()-1);
      int index = int(floor (pointer));
      float remainder = pointer - index;
      retval = ((1-remainder)*histo[index] + remainder*histo[index+1]);
   }
   else if (portion ==1)
      retval = histo[histo.size()-1];
   return retval;
}


void RadialDensitySamplerFunctor::doExecute(
   LensContext *c, const std::vector<DataItem*>& args, 
   std::auto_ptr<DataItem>& rvalue)
{
   ConnectionContext *cc = c->connectionContext;
   ConnectionContext::Responsibility resp = cc->current;
   bool refNodeDifferent = false;
   NodeSet* source = 0;
   NodeDescriptor** slot = 0;

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
	    "RadialDensitySamplerFunctor: invalid responsibility specification");
   }

   if (cc->restart) {
      _sampleSpace.clear();
      std::vector<int> coords, refcoords, begincoords, endcoords, 
	 mincoords, maxcoords, gridSize;
      int radius = int(ceil(_radialScale));
      _refNode->getNodeCoords(refcoords);
      mincoords = source->getBeginCoords();
      maxcoords = source->getEndCoords();
      gridSize =  source->getGrid()->getSize();
      int min, max, minTolerated, maxTolerated, absMax;
      for(unsigned i = 0; i < refcoords.size(); ++i) {
         min = refcoords[i] - radius;
         max = refcoords[i] + radius;
	 minTolerated = mincoords[i] - _borderTolerance;
	 maxTolerated = maxcoords[i] + _borderTolerance;
	 absMax = gridSize[i] - 1;
	 
	 min = (min<0)? 0:min;
	 max = (max>absMax)? absMax:max;

         begincoords.push_back((min<minTolerated) ? minTolerated:min);
         endcoords.push_back((max>maxTolerated)? maxTolerated:max);
      }

      //gather nodes and probabilities
      VolumeOdometer odmtr(begincoords, endcoords);
      float distance, dd;
      for (coords = odmtr.look(); !odmtr.isRolledOver(); 
	   coords = odmtr.next()) {

         distance = 0;
         for(unsigned i = 0; i < coords.size(); ++i) {
            dd = refcoords[i] - coords[i];
            distance += dd*dd;
         }
         distance = sqrt(distance);
         float prob = 0;
         if (distance <=_radialScale) 
	    prob = getRelativeProbability(distance, _radialScale, _list);

         // if probability positive
         if (prob > 0) {
            Sample s;
            std::vector<NodeDescriptor*> nodes;
            NodeSet ns(*source);
            ns.setCoords(coords, coords);
            ns.getNodes(nodes);
            unsigned density =  nodes.size();
            for(unsigned i = 0; i < density; ++i) {
               s.node = nodes[i];
               s.relProb = prob/density;
               _sampleSpace.push_back(s);
            }
         }
      }

      // shuffle
      Sample tmp;
      unsigned pick;
      unsigned top = _sampleSpace.size() -1;
      while (top>0) {
        pick = irandom(0, top, c->sim->getSharedFunctorRandomSeedGenerator());
         tmp = _sampleSpace[top];
         _sampleSpace[top] = _sampleSpace[pick];
         _sampleSpace[pick] = tmp;
         --top;
      };

      // Set up sampling comb
      _probSum=0;
      for (unsigned u = 0; u < _sampleSpace.size(); ++u) {
         _probSum += _sampleSpace[u].relProb;
      }
      _combSpacing = _probSum/(_nbrSamples);
      _combSize = _nbrSamples;
      _combCount = 0;

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
         while (top > 0) {
           pick = irandom(0, top, c->sim->getSharedFunctorRandomSeedGenerator());
            tmp = _sampleSpace[top];
            _sampleSpace[top] = _sampleSpace[pick];
            _sampleSpace[pick] = tmp;
            --top;
         };
         _intervalOffset = _sampleSpace[0].relProb;
         _intervalIndex = 0;

         // Set up sampling comb
         _combSpacing = _probSum / (_nbrSamples - _currentCount);
         _combSize = _nbrSamples - _currentCount;
         _combCount = 0;

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
      _intervalOffset += _sampleSpace[++_intervalIndex].relProb;
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
