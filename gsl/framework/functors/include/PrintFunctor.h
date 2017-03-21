// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef PRINTARGSFUNCTOR_H
#define PRINTARGSFUNCTOR_H
#include "Copyright.h"

#include "Functor.h"
#include <vector>
#include <memory>

class PrintFunctor: public Functor
{
   public:
      PrintFunctor();
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~PrintFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
};
#endif
