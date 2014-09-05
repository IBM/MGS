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
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~BindBackFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:
      Functor *_bind_functor;
      std::vector<DataItem*> _bind_args;
};
#endif
