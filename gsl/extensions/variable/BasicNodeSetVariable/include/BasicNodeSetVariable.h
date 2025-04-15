// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BasicNodeSetVariable_H
#define BasicNodeSetVariable_H
#include "Lens.h"

#include "CG_BasicNodeSetVariable.h"
#include <memory>
#include "ShallowArray.h"

class BasicNodeSetVariable : public CG_BasicNodeSetVariable
{
   public:
      //CUDA_CALLABLE 
      virtual void initialize(RNG&);
      virtual void dca(Trigger* trigger, NDPairList* ndPairList);
      BasicNodeSetVariable();
      virtual ~BasicNodeSetVariable();
      virtual void duplicate(std::unique_ptr<BasicNodeSetVariable>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_BasicNodeSetVariable>& dup) const;
   private:
      ShallowArray<float> _coords;
      int _updateCounter;
      String _generalFileName;
};

#endif
