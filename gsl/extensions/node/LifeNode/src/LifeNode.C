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
//CUDA_CALLABLE 
void LifeNode::initialize(RNG& rng) 
{
   publicValue=value;
}

//CUDA_CALLABLE 
void LifeNode::update(RNG& rng) 
{
   int neighborCount=0;
#if defined(HAVE_GPU) && defined(__NVCC__)
 #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   ShallowArray_Flat<int*>::iterator iter, end = neighbors.end();
   for (iter=neighbors.begin(); iter!=end; ++iter) {
     neighborCount += **iter;
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   auto um_neighbors_from = _container->um_neighbors_start_offset[index];
   auto um_neighbors_to = _container->um_neighbors_num_elements[index]-1;
   for (auto idx = um_neighbors_from; idx < um_neighbors_to; ++idx) {
     neighborCount += *(_container->um_neighbors[idx]);
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   auto um_neighbors_from = index * _container->um_neighbors_max_elements;
   auto um_neighbors_to = _container->um_neighbors_num_elements[index]-1;
   for (auto idx = um_neighbors_from; idx < um_neighbors_to; ++idx) {
     neighborCount += *(_container->um_neighbors[idx]);
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   assert(0);
 #endif
#else
   ShallowArray<int*>::iterator iter, end = neighbors.end();
   for (iter=neighbors.begin(); iter!=end; ++iter) {
     neighborCount += **iter;
   }
#endif
   
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

