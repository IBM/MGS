// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-03-01-2006-1
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef BasicNodeSetVariable_H
#define BasicNodeSetVariable_H
#include "Lens.h"

#include "CG_BasicNodeSetVariable.h"
#include <memory>
#include "ShallowArray.h"

class BasicNodeSetVariable : public CG_BasicNodeSetVariable
{
   public:
      virtual void initialize(RNG&);
      virtual void dca(Trigger* trigger, NDPairList* ndPairList);
      BasicNodeSetVariable();
      virtual ~BasicNodeSetVariable();
      virtual void duplicate(std::auto_ptr<BasicNodeSetVariable>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_BasicNodeSetVariable>& dup) const;
   private:
      ShallowArray<float> _coords;
      int _updateCounter;
      String _generalFileName;
};

#endif
