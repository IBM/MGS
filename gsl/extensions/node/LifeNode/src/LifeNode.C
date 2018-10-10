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

#include "Lens.h"
#include "LifeNode.h"
#include "CG_LifeNode.h"
#include "rndm.h"

CUDA_CALLABLE void LifeNode::initialize(RNG& rng) 
{
   publicValue=value;
}

CUDA_CALLABLE void LifeNode::update(RNG& rng) 
{
   int neighborCount=0;
   ShallowArray<int*>::iterator iter, end = neighbors.end();
   for (iter=neighbors.begin(); iter!=end; ++iter) {
     neighborCount += **iter;
   }
   
   if (neighborCount<= getSharedMembers().tooSparse || neighborCount>=getSharedMembers().tooCrowded) {
     value=0;
   }
   else {
     value=1;
   }
}

void LifeNode::copy(RNG& rng) 
{
  publicValue=value;
}

LifeNode::~LifeNode() 
{
}

