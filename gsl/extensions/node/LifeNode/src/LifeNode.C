#include "Lens.h"
#include "LifeNode.h"
#include "CG_LifeNode.h"
#include "rndm.h"

void LifeNode::initialize(RNG& rng) 
{
   publicValue=value;
}

void LifeNode::update(RNG& rng) 
{
   int neighborCount=0;
   ShallowArray<int*>::iterator iter, end = neightbors.end();
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

