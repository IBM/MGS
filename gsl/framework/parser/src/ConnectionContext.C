// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ConnectionContext.h"

ConnectionContext::ConnectionContext()
   : parent(0), sourceSet(0), destinationSet(0), sourceNode(0), 
     destinationNode(0), sourceRefNode(0), destinationRefNode(0), 
     edgeType(0), edgeInitPSet(0), inAttrPSet(0), outAttrPSet(0), 
     current(_BOTH), currentSample(0), restart(true), done(false)
{
}


ConnectionContext::ConnectionContext(const ConnectionContext& cc)
   : parent(cc.parent), sourceSet(cc.sourceSet), destinationSet(cc.destinationSet),
     sourceNode(cc.sourceNode), destinationNode(cc.destinationNode), sourceRefNode(cc.sourceRefNode),
     destinationRefNode(cc.destinationRefNode), edgeType(cc.edgeType), edgeInitPSet(cc.edgeInitPSet),
     inAttrPSet(cc.inAttrPSet), outAttrPSet(cc.outAttrPSet), 
     current(cc.current), currentSample(cc.currentSample), restart(cc.restart), done(cc.done)
{
}

void ConnectionContext::reset()
{
   parent = 0;
   sourceSet = 0;
   destinationSet = 0;
   sourceNode = 0;
   destinationNode = 0;
   sourceRefNode = 0;
   destinationRefNode = 0;
   edgeType = 0;
   edgeInitPSet = 0;
   inAttrPSet = 0;
   outAttrPSet = 0;
   current = ConnectionContext::_BOTH;
   currentSample = 0;
   restart = true;
   done = false;
}
