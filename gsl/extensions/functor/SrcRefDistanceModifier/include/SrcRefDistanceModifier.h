// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef SrcRefDistanceModifier_H
#define SrcRefDistanceModifier_H
#include "Lens.h"

#include "CG_SrcRefDistanceModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcRefDistanceModifier : public CG_SrcRefDistanceModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcRefDistanceModifier();
      virtual ~SrcRefDistanceModifier();
      virtual void duplicate(std::auto_ptr<SrcRefDistanceModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_SrcRefDistanceModifierBase>& dup) const;
};

#endif
