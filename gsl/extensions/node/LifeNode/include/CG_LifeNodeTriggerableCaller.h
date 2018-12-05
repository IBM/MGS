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

#ifndef CG_LifeNodeTriggerableCaller_H
#define CG_LifeNodeTriggerableCaller_H

#include "Lens.h"
#include "TriggerableCaller.h"
#include <memory>

class CG_LifeNode;

class CG_LifeNodeTriggerableCaller : public TriggerableCaller
{
   public:
      CG_LifeNodeTriggerableCaller(NDPairList* ndPairList, void (CG_LifeNode::*function) (Trigger*, NDPairList*), CG_LifeNode* triggerable);
      virtual void event(Trigger* trigger);
      virtual Triggerable* getTriggerable();
      CG_LifeNodeTriggerableCaller();
      virtual ~CG_LifeNodeTriggerableCaller();
      virtual void duplicate(std::unique_ptr<CG_LifeNodeTriggerableCaller>& dup) const;
      virtual void duplicate(std::unique_ptr<TriggerableCaller>& dup) const;
   private:
      void (CG_LifeNode::*_function) (Trigger*, NDPairList*);
      CG_LifeNode* _triggerable;
};

#endif
