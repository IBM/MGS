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
class LensContext;
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
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~RadialHistoSamplerFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
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
