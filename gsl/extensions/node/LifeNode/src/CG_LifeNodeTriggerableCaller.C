// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CG_LifeNodeTriggerableCaller.h"
#include "CG_LifeNode.h"
#include "TriggerableCaller.h"
#include <memory>

CG_LifeNodeTriggerableCaller::CG_LifeNodeTriggerableCaller(NDPairList* ndPairList, void (CG_LifeNode::*function) (Trigger*, NDPairList*), CG_LifeNode* triggerable) 
   : TriggerableCaller(ndPairList), _function(function), _triggerable(triggerable)
{
}

void CG_LifeNodeTriggerableCaller::event(Trigger* trigger) 
{
   (*_triggerable.*_function)(trigger, _ndPairList);
}

Triggerable* CG_LifeNodeTriggerableCaller::getTriggerable() 
{
   return _triggerable;
}

CG_LifeNodeTriggerableCaller::CG_LifeNodeTriggerableCaller() 
   : TriggerableCaller(), _triggerable(0)
{
}

CG_LifeNodeTriggerableCaller::~CG_LifeNodeTriggerableCaller() 
{
}

void CG_LifeNodeTriggerableCaller::duplicate(std::unique_ptr<CG_LifeNodeTriggerableCaller>& dup) const
{
   dup.reset(new CG_LifeNodeTriggerableCaller(*this));
}

void CG_LifeNodeTriggerableCaller::duplicate(std::unique_ptr<TriggerableCaller>& dup) const
{
   dup.reset(new CG_LifeNodeTriggerableCaller(*this));
}

