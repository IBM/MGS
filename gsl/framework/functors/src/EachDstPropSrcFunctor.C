// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "EachDstPropSrcFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include "FunctorDataItem.h"
#include "NodeSet.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "NumericDataItem.h"
#include "SyntaxErrorException.h"
#include "VectorOstream.h"

EachDstPropSrcFunctor::EachDstPropSrcFunctor()
   : _isUntouched(true), _destinationSet(0), _sourceSet(0), _transforms(0), _toroidalSpacing(0)
{
   _nodesIter = _nodes.begin();
   _nodesEnd = _nodes.end();
}


EachDstPropSrcFunctor::EachDstPropSrcFunctor(const EachDstPropSrcFunctor& csf)
   : _isUntouched(csf._isUntouched), _destinationSet(csf._destinationSet),
     _sourceSet(csf._sourceSet), _nodes(csf._nodes), _slope(csf._slope),
     _transforms(csf._transforms), _coords(csf._coords), _toroidalSpacing(csf._toroidalSpacing)
{
   if (csf._functor_ap.get()) csf._functor_ap->duplicate(std::move(_functor_ap));
   _nodesIter = _nodes.begin();
   _nodesEnd = _nodes.end();
}


void EachDstPropSrcFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new EachDstPropSrcFunctor(*this));
}


EachDstPropSrcFunctor::~EachDstPropSrcFunctor()
{
}


void EachDstPropSrcFunctor::doInitialize(LensContext *c, 
					 const std::vector<DataItem*>& args)
{
  int nbrArgs=args.size();
  if (nbrArgs != 1 && nbrArgs != 2) {
    throw SyntaxErrorException(
			       "Improper number of initialization arguments passed to EachDstPropSrcFunctor");
  }
  FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[0]);
  if (fdi == 0) {
    throw SyntaxErrorException(
			       "Dynamic cast of DataItem to FunctorDataItem failed on EachDstPropSrcFunctor");
  }
  if (fdi->getFunctor()) fdi->getFunctor()->duplicate(std::move(_functor_ap));
  else {
    throw SyntaxErrorException(
			       "Bad functor argument passed to EachDstPropSrcFunctor");
  }
  if (nbrArgs==2) {
    NumericDataItem *toroidalSpacingDI = 
      dynamic_cast<NumericDataItem*>(args[1]);
    if (toroidalSpacingDI==0) {
      std::ostringstream msg;
      msg 
	<< "EachDstPropSrcFunctor: argument 2 is not a NumericDataItem" 
	<< std::endl;
      throw SyntaxErrorException(msg.str());
    }
    _toroidalSpacing=toroidalSpacingDI->getInt();
  }
}


NodeDescriptor* EachDstPropSrcFunctor::getProportionalNode(LensContext *c)
{

//    This method returns a node from the Source nodeset in the 
//    connection context that is in the same relative location within 
//    the source nodeset that the destination node is within the 
//    destination nodeset.
//
//    The following code just gives the destination Node so that the 
//    parser can be tested. The resulting behavior is equivalent to 
//    EachDst. Obviously, the code should be replaced.

   ConnectionContext* cc = c->connectionContext;
   std::vector<int> nodecoords;
   cc->destinationNode->getNodeCoords(nodecoords);
   const std::vector<int>& begincoords = cc->destinationSet->getBeginCoords();
   for(int i = 0; i < _transforms; ++i) {
      _coords[i] = int(_slope[i] * (nodecoords[i] - begincoords[i]) 
		       + cc->sourceSet->getBeginCoords()[i]);
   }
   //if (_toroidalSpacing!=0) std::cerr<<_coords<<" : "<<nodecoords<<std::endl;

   NodeSet sourceCopy(*cc->sourceSet);
   std::vector<int> indices;
   // we need a node at this location for a reference, 
   // set index to zero to avoid unnecessary list building
   indices.push_back(0);
   sourceCopy.setCoords(_coords, _coords);
   sourceCopy.setIndices(indices);
   std::vector<NodeDescriptor*> nodes;
   sourceCopy.getNodes(nodes);
   if (nodes.size() == 0) {
      sourceCopy.setAllLayers();
      sourceCopy.getNodes(nodes);
   }
   if (nodes.size() == 0) {
      // should put grid name and coords in error report
      throw SyntaxErrorException("No reference node at specified coordinate");
   }
   return nodes[0];
}


void EachDstPropSrcFunctor::doExecute(LensContext* c, 
				      const std::vector<DataItem*>& args, 
				      std::unique_ptr<DataItem>& rvalue)
{
   ConnectionContext* cc = c->connectionContext;
   bool originalRestart = cc->restart;
   cc->done = false;
   if (cc->restart) {
      _destinationSet = cc->destinationSet;
      _sourceSet = cc->sourceSet;
      _nodes.clear();
      cc->destinationSet->getNodes(_nodes);
      _nodesIter = _nodes.begin();
      _nodesEnd = _nodes.end();
      _isUntouched = false;

      // determine dimensions of source and destination sets
      const std::vector<int>& sourceCoords = _sourceSet->getEndCoords();
      const std::vector<int>& destCoords = _destinationSet->getEndCoords();
      int sourceDims = sourceCoords.size();
      int destDims = destCoords.size();
      _transforms = (sourceDims < destDims) ? sourceDims : destDims;

      _coords.clear();
      // set _coords to 0 so that non-transformed coordinates will be zero
      for(int i = 0; i < sourceDims; ++i) {
         _coords.push_back(0);
      }

      // compute slopes and offsets
      _slope.clear();
      _slope.resize(_transforms);
      int destSize, sourceSize;
      for (int i = 0; i < _transforms; ++i) {
         destSize = cc->destinationSet->getEndCoords()[i] 
	    - cc->destinationSet->getBeginCoords()[i];
         sourceSize = cc->sourceSet->getEndCoords()[i] 
	    - cc->sourceSet->getBeginCoords()[i];
	 if (destSize==0 && sourceSize==0)  _slope[i]=1.0;
         else if (_toroidalSpacing>0) _slope[i] = double(sourceSize+1)/double(destSize+1);
	 else _slope[i] = double(sourceSize)/double(destSize);
      }
   }

   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   cc->destinationNode = (*_nodesIter);
   cc->sourceRefNode = getProportionalNode(c);
   cc->current = ConnectionContext::_SOURCE;
   _functor_ap->execute(c, nullArgs, rval_ap);
   while(cc->done && _nodesIter!=_nodesEnd) {
      cc->done = false;
      ++_nodesIter;
      if (_nodesIter == _nodesEnd) {
	 break;
      }
      cc->destinationNode = (*_nodesIter);
      cc->sourceRefNode = getProportionalNode(c);
      cc->current = ConnectionContext::_SOURCE;
      cc->restart = true;
      _functor_ap->execute(c, nullArgs, rval_ap);
      cc->restart = originalRestart;
   }
   if (_nodesIter == _nodesEnd) {
      cc->destinationNode = 0;
      cc->sourceNode = 0;
      cc->done = true;
   }
   cc->restart = originalRestart;
}


/* ***********************
Grab currentSample number
if (first time){
Grab list of destination nodes from the destination nodeset (in a list passed to the NodeSet object)
Set iterator to begin
}

set SourceReferencePoint to Node from iterator
set Destination node from iterator
call SamplingFctr1
if (source node is 0)
{
increment  iterator
if (iterator == end) return 0 as source and destination nodes
}
return
* ************************/

/* ********************* *
 check whether it's SOURCE or DEST,
 check that the count of generated
 nodes didn't reach limit,
 check that reference node is set
 and then generate a list of nodes
 within the ring and pick one of them,
 otherwise just pick one of them
 (uniform distribuition),
 then update the count of generated
 nodes.
* ********************* */
