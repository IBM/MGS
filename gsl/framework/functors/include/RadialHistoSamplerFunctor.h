// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _RADIALHISTOSAMPLERFUNCTOR_H_
#define _RADIALHISTOSAMPLERFUNCTOR_H_
#include "Copyright.h"

#include "SampFctr1Functor.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include <memory>
#include <list>
#include <vector>

class DataItem;
class GslContext;
class NodeSet;

class RadialHistoSamplerFunctor: public SampFctr1Functor
{
   public:
      class Sample
      {
         public:
 	    Sample() 
	      : node(0), relProb(0), distance(0) {}
	    Sample(const Sample& s) 
	      : node(s.node), relProb(s.relProb), distance(s.distance) {}
            NodeDescriptor *node;
            float relProb;
            float distance;
            bool operator<(Sample const &s) const
            {
               return distance< s.distance;
            }
	    ~Sample() {}
      };

      class CummulativeProbabilitySegment
      {
         public:
            float prevCummulative;
            float slope;
            float offset;
      };

      RadialHistoSamplerFunctor();
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~RadialHistoSamplerFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
      float getCummulativeProbability(
	 float distance, float scale, 
	 std::vector<CummulativeProbabilitySegment> &segs);
   private:
      unsigned _nbrSamples;
      float _radialScale;
      unsigned _borderTolerance;
      std::vector<NodeDescriptor*> _sampleSet;
      std::vector<float> _list;
      ConnectionContext::Responsibility _responsibility;
      int _currentSample;
      NodeDescriptor *_refNode;
      unsigned _currentCount;
      unsigned _combCount;
      double _combSpacing;
      unsigned _combSize;
      unsigned _intervalIndex;
      double _intervalOffset;
      std::vector<Sample> _sampleSpace;
      std::vector<CummulativeProbabilitySegment> _cummProbSeg;
      double _probSum;
};
#endif
