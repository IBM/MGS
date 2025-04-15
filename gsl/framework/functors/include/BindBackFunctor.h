// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _BIND_BACK_FUNCTOR_H_
#define _BIND_BACK_FUNCTOR_H_
#include "Copyright.h"

#include <string>
#include <vector>
#include <memory>
#include "Functor.h"

class BindBackFunctor: public Functor
{
   public:
      BindBackFunctor();
      BindBackFunctor(const BindBackFunctor &);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~BindBackFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      Functor *_bind_functor;
      std::vector<DataItem*> _bind_args;
};
#endif
