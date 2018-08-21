// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef TissueLayoutFunctor_H
#define TissueLayoutFunctor_H

#include "Lens.h"
#include "CG_TissueLayoutFunctorBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include "TissueElement.h"
#include <memory>

class TissueFunctor;

class TissueLayoutFunctor : public CG_TissueLayoutFunctorBase, public TissueElement
{
   public:
      void userInitialize(LensContext* CG_c);
      ShallowArray< int > userExecute(LensContext* CG_c);
      TissueLayoutFunctor();
      TissueLayoutFunctor(TissueLayoutFunctor const &);
      virtual ~TissueLayoutFunctor();
      virtual void duplicate(std::auto_ptr<TissueLayoutFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_TissueLayoutFunctorBase>& dup) const;
      void setTissueFunctor(TissueFunctor* tf) {_tissueFunctor=tf;}

   private:
      TissueFunctor* _tissueFunctor;
};

#endif
