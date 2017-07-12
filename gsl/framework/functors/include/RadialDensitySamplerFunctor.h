// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef _RADIALDENSITYSAMPLERFUNCTOR_H_
#define _RADIALDENSITYSAMPLERFUNCTOR_H_
#include "Copyright.h"

#include "SampFctr1Functor.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include "RNG.h"
#include <memory>
#include <list>
#include <vector>

class DataItem;
class LensContext;
class NodeSet;

class RadialDensitySamplerFunctor: public SampFctr1Functor
{
   public:
      class Sample
      {
         public:
	    Sample() 
	      : node(0), relProb(0), valid(false) {}
	    Sample(const Sample& s) 
	      : node(s.node), relProb(s.relProb), valid(s.valid) {}
            NodeDescriptor *node;
            float relProb;
            bool valid;
	    ~Sample() {}
      };

      RadialDensitySamplerFunctor();
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~RadialDensitySamplerFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
      float getRelativeProbability(float distance, float scale, 
				   std::vector<float> &histogram);
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
      double _probSum;
};
#endif
