// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "LifeNode.h"
#include "CG_LifeNode.h"
#if defined(HAVE_GPU) 
#include "CG_LifeNodeCompCategory.h"
#endif
#include "rndm.h"

#define SHD getSharedMembers()

#if defined(HAVE_GPU) 
#define value (_container->um_value[index]) 
#define publicValue (_container->um_publicValue[index]) 
#define neighbors (_container->um_neighbors[index]) 
#define weight (_container->um_weight[index]) 
#define publicWeight  (_container->um_publicWeight[index])
#define neighborsWeight  (_container->um_neighborsWeight[index])
#endif
void LifeNode::initialize(RNG& rng) 
{
   publicValue=value;
   weight = drandom(-1,1, rng);
}

void LifeNode::update(RNG& rng) 
{
   int neighborCount=0;
   auto end = neighbors.end();
   for (auto iter=neighbors.begin(); iter!=end; ++iter) {
     neighborCount += **iter;
   }
   
   if (neighborCount<= getSharedMembers().tooSparse || neighborCount>=getSharedMembers().tooCrowded) {
     value=0;
   }
   else {
     value=1;
   }
   /* reproduction */
   //if (neighborCount == 3 and value == 0) value = 1;
}
//void LifeNode::update(RNG& rng) 
//{
//#if defined(HAVE_GPU) 
//   {
// #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
//   auto end = neighbors.end();
//   for (auto iter=neighbors.begin(); iter!=end; ++iter) {
//     neighborCount += **iter;
//   }
//   /* NOT recommended
//   ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UNIFIED_MEM>::iterator iter, end = neighbors.end();
//   for (iter=neighbors.begin(); iter!=end; ++iter) {
//     neighborCount += **iter;
//   }
//   */
// #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
//   /* NOTE: *iter
//    * becomes
//    *       (_container->um_neighbors[idx])
//    */
//   auto um_neighbors_from = _container->um_neighbors_start_offset[index];
//   auto um_neighbors_to = _container->um_neighbors_num_elements[index]-1;
//   for (auto idx = um_neighbors_from; idx < um_neighbors_to; ++idx) {
//     neighborCount += *(_container->um_neighbors[idx]);
//   }
// #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
//   auto um_neighbors_from = index * _container->um_neighbors_max_elements;
//   auto um_neighbors_to = um_neighbors_from + _container->um_neighbors_num_elements[index];
//   for (auto idx = um_neighbors_from; idx < um_neighbors_to; ++idx) {
//     neighborCount += *(_container->um_neighbors[idx]);
//   }
// #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
//   assert(0);
// #endif
//   }
//#else
//   //original code [before GPU-support]
//   ShallowArray<int*>::iterator iter, end = neighbors.end();
//   for (iter=neighbors.begin(); iter!=end; ++iter) {
//     neighborCount += **iter;
//   }
//#endif
//}

void LifeNode::updateWeight(RNG& rng) 
{
  float weightSum = 0;
  float dw = 0;
  // add your code here
  for (int ii = 0; ii < SHD.complexity; ii++)
  {
#if defined(HAVE_GPU) 
    {
  #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
      auto end = neighborsWeight.end();
      for (auto iter=neighborsWeight.begin(); iter!=end; ++iter) {
        weightSum += **iter;
      }
  #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      auto um_neighborsWeight_from = index * _container->um_neighborsWeight_max_elements;
      auto um_neighborsWeight_to = um_neighborsWeight_from + _container->um_neighborsWeight_num_elements[index];
      for (auto idx = um_neighborsWeight_from; idx < um_neighborsWeight_to; ++idx) {
        weightSum += *(_container->um_neighborsWeight[idx]);
      }
  #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
  #endif
    }
#else 
   //original code [before GPU-support]
   ShallowArray<float*>::iterator iter, end = neighborsWeight.end();
   for (iter=neighborsWeight.begin(); iter!=end; ++iter) {
     weightSum += **iter; 
   }
#endif
    if (SHD.actionType == F_SIGMOID)
      dw += sigmoid(weightSum);
    else if (SHD.actionType == F_ReLU)
      dw += ReLU(weightSum);
    else if (SHD.actionType == F_TANH)
      dw += tanh(weightSum);

    if (value == 1) {
      weight += dw;
    }
    else {
      weight -= dw;
    }
  }
}

void LifeNode::copy(RNG& rng) 
{
  publicValue = value;
  publicWeight = weight;
}

LifeNode::~LifeNode() 
{
}

