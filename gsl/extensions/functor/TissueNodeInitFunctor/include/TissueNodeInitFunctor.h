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

#ifndef TissueNodeInitFunctor_H
#define TissueNodeInitFunctor_H

#include "Lens.h"
#include "CG_TissueNodeInitFunctorBase.h"
#include "LensContext.h"
#include "TissueElement.h"
#include <memory>

class TissueNodeInitFunctor : public CG_TissueNodeInitFunctorBase, public TissueElement
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c);
      TissueNodeInitFunctor();
      TissueNodeInitFunctor(TissueNodeInitFunctor const &);
      virtual ~TissueNodeInitFunctor();
      virtual void duplicate(std::auto_ptr<TissueNodeInitFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_TissueNodeInitFunctorBase>& dup) const;
      void setTissueFunctor(TissueFunctor* tf) {_tissueFunctor=tf;}

   private:
      TissueFunctor* _tissueFunctor;
};

#endif
