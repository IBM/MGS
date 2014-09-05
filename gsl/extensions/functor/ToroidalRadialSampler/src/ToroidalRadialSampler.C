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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "ToroidalRadialSampler.h"
#include "CG_ToroidalRadialSamplerBase.h"
#include "LensContext.h"
#include "ConnectionContext.h"
#include "ParameterSet.h"
#include "NodeSet.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "VectorOstream.h"
#include <memory>
#include <vector>
#include <list>
#include <cmath>

void ToroidalRadialSampler::userInitialize(LensContext* CG_c, float& radius)
{
}

void ToroidalRadialSampler::userExecute(LensContext* CG_c) 
{
   ConnectionContext *cc = CG_c->connectionContext;
   ConnectionContext::Responsibility resp = cc->current;
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
          }
          break;
      case ConnectionContext::_DEST:
         //if (_speak) std::cout<<" each destination node based on a complete sampling within a radius surrounding a ref node";
         source = cc->destinationSet;
         slot = &cc->destinationNode;
         if(_refNode != cc->destinationRefNode) {
            _refNode = cc->destinationRefNode;
	    _refNode->getNodeCoords(_refcoords);
         }
         break;
      case ConnectionContext::_BOTH:
         throw SyntaxErrorException(
	    "RadialSamplerFunctor: invalid responsibility specification");
   }

   if (cc->restart) {
     /*
     static std::vector<int> checkCoords;
     if (checkCoords.size() != 0) {
       bool allEq = true;
       for (int i = 0; i < checkCoords.size(); ++i) {
	 if (checkCoords[i] != _refcoords[i]) {
	   allEq = false;
	   break;
	 }
       }
       if (allEq) {
	 std::cerr<<_refcoords<<std::endl;
	 exit(-1);
       }
     }
     checkCoords = _refcoords;
     */

     _nodes.clear();
     _nodeSetSize.clear();
     std::vector<int> begincoords, endcoords, mincoords, maxcoords;
     std::list<std::pair<std::vector<int>, std::vector<int> > > stack;
     mincoords = source->getBeginCoords();
     maxcoords = source->getEndCoords();
     unsigned dims = _refcoords.size();
     for(unsigned i=0;i<dims;++i) {
       begincoords.push_back(_refcoords[i] - int(ceil(init.radius)));
       endcoords.push_back(_refcoords[i] + int(ceil(init.radius)));
       _nodeSetSize.push_back(maxcoords[i]-mincoords[i]);
     }
     std::pair<std::vector<int>, std::vector<int> > element(begincoords, endcoords);
     stack.push_front(element);
     //std::cerr<<"stack in : "<<begincoords<<" | "<<endcoords<<std::endl;
     while (stack.size()>0) {
       element=*(stack.begin());
       stack.pop_front();
       begincoords=element.first;
       endcoords=element.second;
       //std::cerr<<"stack out : "<<begincoords<<" | "<<endcoords<<std::endl;
       unsigned i=0;
       for(;i<dims;++i) {
	 if (begincoords[i]<mincoords[i] && endcoords[i]>maxcoords[i]) {
	   begincoords[i]=mincoords[i];
	   endcoords[i]=maxcoords[i];
	 }
	 else {
	   if (begincoords[i]<mincoords[i]) {
	     std::pair<std::vector<int>, std::vector<int> > element2(element);	     
	     element.first[i]=mincoords[i];
	     element2.first[i]=maxcoords[i]-(mincoords[i]-begincoords[i])+1;
	     element2.second[i]=maxcoords[i];
	     stack.push_front(element);
	     stack.push_front(element2);
 	     //std::cerr<<"stack in : "<<element.first<<" | "<<element.second<<std::endl;
	     //std::cerr<<"stack in : "<<element2.first<<" | "<<element2.second<<std::endl;
	     break;
	   }
	   if (endcoords[i]>maxcoords[i]) {
	     std::pair<std::vector<int>, std::vector<int> > element2(element);	         
	     element.second[i]=maxcoords[i];
	     element2.first[i]=mincoords[i];
	     element2.second[i]=mincoords[i]+(endcoords[i]-maxcoords[i])-1;
	     stack.push_front(element);
	     stack.push_front(element2);
 	     //std::cerr<<"stack in : "<<element.first<<" | "<<element.second<<std::endl;
	     //std::cerr<<"stack in : "<<element2.first<<" | "<<element2.second<<std::endl;
	     break;
	   }
	 }
       }
       if (i==dims) {
	 NodeSet ns(*source);
	 ns.setCoords(begincoords, endcoords);
	 std::vector<NodeDescriptor*> nodes;
	 ns.getNodes(nodes);	  
	 std::vector<NodeDescriptor*>::iterator nodesIter, nodesEnd=nodes.end();
	 for (nodesIter=nodes.begin(); nodesIter!=nodesEnd; ++nodesIter) _nodes.push_back(*nodesIter);
	 //std::cerr<<_nodes.size()<<std::endl;
       }
     }
     _currentNode = 0;
      _nbrNodes = _nodes.size();
   } // end of if (cc->restart)

   if (_currentNode==_nbrNodes) {
     *slot = 0;
     cc->done = true;
     //std::cerr<<"done1:"<<_currentNode<<std::endl;
     return;
   }
   else {
     float distance=0, ddist;
     bool outside=true;
     
     std::vector<int>::iterator i1, i2, i3, end1;
     NodeDescriptor* n;
     std::vector<int> coords;

     while (outside) {
       n = _nodes[_currentNode];
       n->getNodeCoords(coords);
       end1 = coords.end();
       distance = 0;
       //std::cerr<<"coords: "<<coords<<" _refcoords:"<<_refcoords<<std::endl;
       for (i1=coords.begin(),i2=_refcoords.begin(),i3=_nodeSetSize.begin();i1!=end1;++i1, ++i2, ++i3){
	 ddist = abs(*i1 - *i2);
	 if (ddist > *i3/2) ddist = *i3 - ddist;
	 distance += ddist*ddist;
       }
       distance=sqrt(distance);

       //std::cerr<<distance<<"<"<<init.radius<<"?"<<std::endl;
       if (distance<=init.radius) {
	 outside=false;
	 *slot = _nodes[_currentNode];
	 //std::cerr<<"chosen:"<<_currentNode<<std::endl;
       }
       else if (++_currentNode==_nbrNodes) {
	 *slot = 0;
	 cc->done = true;
	 //std::cerr<<"done2:"<<_currentNode<<std::endl;
	 return;
       }
     }
     ++_currentNode;
     cc->done = false;
   }
}

ToroidalRadialSampler::ToroidalRadialSampler() 
   : CG_ToroidalRadialSamplerBase()
{
}

ToroidalRadialSampler::~ToroidalRadialSampler() 
{
}

void ToroidalRadialSampler::duplicate(std::auto_ptr<ToroidalRadialSampler>& dup) const
{
   dup.reset(new ToroidalRadialSampler(*this));
}

void ToroidalRadialSampler::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new ToroidalRadialSampler(*this));
}

void ToroidalRadialSampler::duplicate(std::auto_ptr<CG_ToroidalRadialSamplerBase>& dup) const
{
   dup.reset(new ToroidalRadialSampler(*this));
}

