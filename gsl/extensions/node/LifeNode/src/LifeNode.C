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
#if defined(HAVE_GPU) && defined(__NVCC__)
#include "CG_LifeNodeCompCategory.h"
#endif
#include "rndm.h"

#if defined(HAVE_GPU) && defined(__NVCC__)
#define value (_container->um_value[index]) 
#define publicValue (_container->um_publicValue[index]) 
#define neighbors (_container->um_neighbors[index]) 
#endif
CUDA_CALLABLE void LifeNode::initialize(RNG& rng) 
{
   publicValue=value;
}

CUDA_CALLABLE void LifeNode::update(RNG& rng) 
{
   int neighborCount=0;
#if defined(HAVE_GPU) && defined(__NVCC__)
   ShallowArray_Flat<int*>::iterator iter, end = neighbors.end();
#else
   ShallowArray<int*>::iterator iter, end = neighbors.end();
#endif
   for (iter=neighbors.begin(); iter!=end; ++iter) {
     neighborCount += **iter;
   }
   
   //TUAN TODO 
   /// consider here as we cannot access SharedMembers directly
   // maybe for all shared member, we should pass via argument?
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

