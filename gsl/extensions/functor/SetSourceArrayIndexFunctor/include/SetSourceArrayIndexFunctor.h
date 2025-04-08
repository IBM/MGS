#ifndef SetSourceArrayIndexFunctor_H
#define SetSourceArrayIndexFunctor_H

#include "Lens.h"
#include "CG_SetSourceArrayIndexFunctorBase.h"
#include "LensContext.h"
#include <memory>
#include <map>

class NodeDescriptor;

class SetSourceArrayIndexFunctor : public CG_SetSourceArrayIndexFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& destinationInAttr);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SetSourceArrayIndexFunctor();
      SetSourceArrayIndexFunctor(SetSourceArrayIndexFunctor const&);
      virtual ~SetSourceArrayIndexFunctor();
      virtual void duplicate(std::unique_ptr<SetSourceArrayIndexFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SetSourceArrayIndexFunctorBase>&& dup) const;
   private:
      std::map<NodeDescriptor*, unsigned> _indexMap;
      std::unique_ptr<Functor> _destinationInAttr;
};

#endif
