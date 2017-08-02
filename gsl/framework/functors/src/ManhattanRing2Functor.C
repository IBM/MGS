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

#include "ManhattanRing2Functor.h"
#include "LensContext.h"
#include "ConnectionContext.h"
#include "DataItem.h"
#include "IntArrayDataItem.h"
#include "FunctorType.h"
#include "GridLayerDescriptor.h"
#include "SurfaceOdometer.h"
#include "Grid.h"
#include "Simulation.h"
#include "NodeDescriptor.h"
#include "NodeSet.h"
#include "NodeAccessor.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "rndm.h"
#include "SyntaxErrorException.h"
#include <stdlib.h>
#include <sstream>

ManhattanRing2Functor::ManhattanRing2Functor()
: _sampleSet(), _list(),_responsibility(ConnectionContext::_BOTH),
_currentSample(0), _refNode(0), _currentCount(0),_currentList(0)
{
}

void ManhattanRing2Functor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new ManhattanRing2Functor(*this));
}


ManhattanRing2Functor::~ManhattanRing2Functor()
{
}


void ManhattanRing2Functor::doInitialize(LensContext *c, 
					 const std::vector<DataItem*>& args)
{
   /*
   Grab the int list
   Copy and store
   Set up current count lists to initial condition
   */

   // prototype SamplingFctr2 ManhattanRing2(std::list<int>,int direction);

   IntArrayDataItem *ia_di = dynamic_cast<IntArrayDataItem*>(args[0]);
   if (ia_di==0) {
      throw SyntaxErrorException(
	 "ManhattanRing2Functor: expected list of ints in initialization");
   }
   _list = *ia_di->getIntVector();
}


void ManhattanRing2Functor::doExecute(LensContext *c, 
				      const std::vector<DataItem*>& args, 
				      std::auto_ptr<DataItem>& rvalue)
{
   ConnectionContext *cc = c->connectionContext;
   ConnectionContext::Responsibility resp = cc->current;
   bool refNodeDifferent = false;
   NodeSet *source = 0;
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
	    "ManhattanRing2Functor: invalid responsibility specification");
   }

   if (cc->restart) {
      std::vector<int> coords;
      _refNode->getNodeCoords(coords);
      collectRadialNodes(coords, source, _list, _sampleSet);
      if (_list.size() != _sampleSet.size()) {
         std::ostringstream msg;
	 msg << "ManhattanRing2: mismatch in size of radius list and sample set: "
	     << _list.size() << ", " << _sampleSet.size() << std::endl;
	 throw SyntaxErrorException(msg.str());
      }
      _responsibility = resp;
      _currentCount = 0;
      _currentList = 0;
      _currentSample = cc->currentSample;
   }

   if (_currentSample != cc->currentSample 
       || _currentCount>=unsigned(_list[_currentList])) {
      _currentSample = cc->currentSample;
      ++_currentCount;
      if (_currentCount >= unsigned(_list[_currentList])) {
         ++_currentList;
         _currentCount =0;
      }

   }
   while (_currentList < _list.size() && _list[_currentList] == 0) {
      ++_currentList;
   }
   if (_currentList >= _list.size()) {
      *slot = 0;
      cc->done = true;
      return;
   }

   // sample, set, and return
   std::vector<NodeDescriptor*>& current = _sampleSet[_currentList];
   if (current.size() == 0) {
      throw SyntaxErrorException(
	 "ManhattanRing2: No nodes available at requested radius");
   }
   *slot = current[irandom(0, current.size() - 1,c->sim->getSharedFunctorRandomSeedGenerator())];
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

void ManhattanRing2Functor::collectRadialNodes(
   std::vector<int> origin, NodeSet* sourceSet, 
   std::vector<int> & radiusSample, 
   std::vector<std::vector<NodeDescriptor*> >& collectedRadialNodes)
{
   collectedRadialNodes.clear();
   const std::vector<int>& sourceBegin = sourceSet->getBeginCoords();
   const std::vector<int>& sourceEnd = sourceSet->getEndCoords();
   unsigned dims = sourceEnd.size();
   int nbrRadii = radiusSample.size();

   if (origin.size() != dims) {
      throw SyntaxErrorException("ManhattanRing2: the origin coordinates are incompatible with the NodeSet dimensions");
   }
   std::vector<int> begin, end;

   for (int radius = 0; radius < nbrRadii; radius++) {

      std::vector<NodeDescriptor*> radiusVec; // holds handles for this radius
      radiusVec.clear();
      if (radiusSample[radius]) {
         begin = origin;
         end = origin;
         for (unsigned int i = 0; i<dims; i++) {
            begin[i]-=radius;
            end[i]+=radius;
         }

         SurfaceOdometer odmtr(begin, end);
         std::vector<int> & pcoord = odmtr.look();

         // make a copy of pcoord so the original
         // is not modified
         int sz = pcoord.size();
         std::vector<int> pcoord2(sz);

         for (; !odmtr.isRolledOver(); odmtr.next() ) {
            bool modified = false;

            for (int idx=dims-1; idx>=0; idx--) {
               pcoord2[idx] = pcoord[idx];
               if ((pcoord[idx]>sourceEnd[idx])) {
                  modified = true;
                  pcoord2[idx] = sourceBegin[idx] + (pcoord[idx] - sourceEnd[idx]) - 1;

               }
               else
               if( pcoord[idx]<sourceBegin[idx]) {
                  modified = true;
                  pcoord2[idx] = sourceEnd[idx] - (sourceBegin[idx] - pcoord[idx]) + 1;
               }
            }

            if(modified) {
               // use pcoord2, the modified coordinates
               NodeSet local(*sourceSet);
               local.setCoords(pcoord2, pcoord2);
               std::vector<NodeDescriptor*> nodes;
               local.getNodes(nodes);
               for(unsigned int n=0;n<nodes.size();++n)
                  radiusVec.push_back(nodes[n]);
            }
            else {
               // use pcoord, the original coordinates
               NodeSet local(*sourceSet);
               local.setCoords(pcoord, pcoord);
               std::vector<NodeDescriptor*> nodes;
               local.getNodes(nodes);
               for(unsigned int n=0;n<nodes.size();++n)
                  radiusVec.push_back(nodes[n]);
            }
         }
      }                          // of if radiusSample[]
      // put this radius' handle on the objects vector
      /*
            std::cerr << "ManhattanRing2: nodes="<<radiusVec.size()<<", r="<<radius;
            std::cerr <<", origin=";
            char token = '[';
            for (int i = 0;i<origin.size();i++) {
              std::cerr <<token<<origin[i];
              if (token == '[') token = ',';
            }
            std::cerr <<"], sourceSet=("<<*sourceSet<<")"<<std::endl;
      */
      collectedRadialNodes.push_back(radiusVec);
   }
}
