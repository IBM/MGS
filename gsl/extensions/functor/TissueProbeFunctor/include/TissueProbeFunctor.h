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

#ifndef TissueProbeFunctor_H
#define TissueProbeFunctor_H

#include "Lens.h"
#include "CG_TissueProbeFunctorBase.h"
#include "LensContext.h"
#include "NodeSet.h"
#include "TissueElement.h"
#include <memory>

class TissueProbeFunctor : public CG_TissueProbeFunctorBase, public TissueElement
{
   public:
      void userInitialize(LensContext* CG_c);
      std::auto_ptr<NodeSet> userExecute(LensContext* CG_c);
      TissueProbeFunctor();
      virtual ~TissueProbeFunctor();
      virtual void duplicate(std::auto_ptr<TissueProbeFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_TissueProbeFunctorBase>& dup) const;
      void setTissueFunctor(TissueFunctor* tf) {_tissueFunctor=tf;}

   private:
      TissueFunctor* _tissueFunctor;
};

#endif
