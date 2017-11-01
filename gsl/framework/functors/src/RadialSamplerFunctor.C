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

#include "RadialSamplerFunctor.h"
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
#include "NodeDescriptor.h"
#include "NodeSet.h"
#include "NodeAccessor.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"
#include <sstream>
#include <stdlib.h>
#include <math.h>

RadialSamplerFunctor::RadialSamplerFunctor()
: _responsibility(ConnectionContext::_BOTH), _refNode(0), _radius(0), 
  _borderTolerance(0), _direction(0), _currentNode(0), _nbrNodes(0)
{
}

RadialSamplerFunctor::RadialSamplerFunctor(const RadialSamplerFunctor& rsf)
: _responsibility(rsf._responsibility), _refNode(rsf._refNode), _radius(rsf._radius), 
  _borderTolerance(rsf._borderTolerance), _direction(rsf._direction),
  _currentNode(rsf._currentNode), _nbrNodes(rsf._nbrNodes)
{
  _nodes=rsf._nodes;
  _refcoords=rsf._refcoords;
}

void RadialSamplerFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new RadialSamplerFunctor(*this));
}


RadialSamplerFunctor::~RadialSamplerFunctor()
{
}


void RadialSamplerFunctor::doInitialize(LensContext *c, 
					const std::vector<DataItem*>& args)
{
  int nbrArgs=args.size();
  if (nbrArgs!=1 && nbrArgs !=2 && nbrArgs !=3) {
      std::ostringstream msg;
      msg << "RadialSampler: invalid arguments!" << std::endl
	  << "\texpected: RadialSampler(float radius) or" << std::endl
	  << "\texpected: RadialSampler(float radius, int borderTolerance)" << std::endl      
	  << "\texpected: RadialSampler(float radius, int borderTolerance, int direction)" 
	  << std::endl;
      throw SyntaxErrorException(msg.str());
  }
  NumericDataItem *radiusDI = dynamic_cast<NumericDataItem*>(args[0]);
  if (radiusDI==0) {
    std::ostringstream msg;
    msg << "RadialSampler: argument 1 is not a NumericDataItem" << std::endl
	<< "\texpected: RadialSampler(float radius)" << std::endl
	<< "\texpected: RadialSampler(float radius, int borderTolerance)" 
	<< std::endl;
    throw SyntaxErrorException(msg.str());
  }
  _radius=radiusDI->getFloat();
  if (nbrArgs==2) {
    NumericDataItem *borderToleranceDI = 
      dynamic_cast<NumericDataItem*>(args[1]);
    if (borderToleranceDI==0) {
      std::ostringstream msg;
      msg 
	<< "RadialSampler: argument 2 is not a NumericDataItem" 
	<< std::endl
	<< "\texpected: RadialSampler(float radius, int borderTolerance)."
	<< std::endl;
      throw SyntaxErrorException(msg.str());
    }
    _borderTolerance=unsigned(borderToleranceDI->getInt());
  }
  if (nbrArgs==3) {
    NumericDataItem *directionDI = 
      dynamic_cast<NumericDataItem*>(args[2]);
    if (directionDI==0) {
      std::ostringstream msg;
      msg 
        << "RadialSampler: argument 3 is not a NumericDataItem" 
        << std::endl
        << "\texpected: RadialSampler(float radius, int borderTolerance, int direction)."
        << std::endl;
      throw SyntaxErrorException(msg.str());
    }    
    _direction=unsigned(directionDI->getInt());
  }
}

void RadialSamplerFunctor::doExecute(LensContext *c, 
				     const std::vector<DataItem*>& args, 
				     std::auto_ptr<DataItem>& rvalue)
{
   ConnectionContext *cc = c->connectionContext;
   ConnectionContext::Responsibility resp = cc->current;
   bool refNodeDifferent = false;
   NodeSet* source=0;
   NodeDescriptor** slot=0;

   switch(resp) {
      case ConnectionContext::_SOURCE:
         //if (_speak) std::cout<<" each source node based on a complete sampling with a radius surrounding a ref node";
         source = cc->sourceSet;
         slot = &cc->sourceNode;
         if(_refNode != cc->sourceRefNode) {
            _refNode = cc->sourceRefNode;
	    _refNode->getNodeCoords(_refcoords);
            refNodeDifferent = true;
         }
         break;
      case ConnectionContext::_DEST:
         //if (_speak) std::cout<<" each destination node based on a complete sampling within a radius surrounding a ref node";
         source = cc->destinationSet;
         slot = &cc->destinationNode;
         if(_refNode != cc->destinationRefNode) {
            _refNode = cc->destinationRefNode;
	    _refNode->getNodeCoords(_refcoords);
            refNodeDifferent = true;
         }
         break;
      case ConnectionContext::_BOTH:
         throw SyntaxErrorException(
	    "RadialSamplerFunctor: invalid responsibility specification");
   }

   if (cc->restart) {
     std::vector<int> coords, begincoords, endcoords, 
       mincoords, maxcoords, gridSize;
     mincoords = source->getBeginCoords();
     maxcoords = source->getEndCoords();
     gridSize =  source->getGrid()->getSize();
     int min, max, minTolerated, maxTolerated, absMax;
     for(unsigned i=0;i<_refcoords.size();++i) {
       min = _refcoords[i] - int(ceil(_radius));
       max = _refcoords[i] + int(ceil(_radius));
       minTolerated = mincoords[i]-_borderTolerance;
       maxTolerated = maxcoords[i]+_borderTolerance;
       absMax = gridSize[i]-1;

       min = (min<0)? 0:min;
       max = (max>absMax)? absMax:max;
       begincoords.push_back((min<minTolerated)? minTolerated:min);
       endcoords.push_back((max>maxTolerated)? maxTolerated:max);
     }

     NodeSet ns(*source);
     ns.setCoords(begincoords, endcoords);
     ns.getNodes(_nodes);
     _currentNode = 0;
     _nbrNodes = _nodes.size();
   }
    
   if (_currentNode==_nbrNodes) {
     *slot = 0;
     cc->done = true;
     return;
   }
   
   float distance, dd;
   bool outside=true;
   NodeDescriptor* n;
   std::vector<int> coords;
   while (outside) {
     n = _nodes[_currentNode];
     n->getNodeCoords(coords);     
     if (
         (_direction == 0) || // both direction
         ((_direction > 0) && // positive direction
          ((coords[0] >= _refcoords[0])
           && (coords[1] >= _refcoords[1])
           && (coords[2] >= _refcoords[2]))) ||
         ((_direction < 0) && // negative direction
          ((coords[0] <= _refcoords[0])
           && (coords[1] <= _refcoords[1])
           && (coords[2] <= _refcoords[2])))
         )
       {
         
         distance = 0;
         for(unsigned i=0;i<coords.size();++i) {
           dd = _refcoords[i] - coords[i];
           distance += dd*dd;
         }
         distance=sqrt(distance);

       }
     else
       distance = _radius + 1.0;
     
     if (distance<=_radius) {
       outside=false;
       *slot = _nodes[_currentNode];
      }
     else if (++_currentNode==_nbrNodes) {
       *slot = 0;
       cc->done = true;
       return;
     }
   }
   ++_currentNode;
   cc->done = false;
}
