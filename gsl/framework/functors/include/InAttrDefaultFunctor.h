// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef _INATTRDEFAULTFUNCTOR_H_
#define _INATTRDEFAULTFUNCTOR_H_
#include "Copyright.h"

#include <memory>
#include <list>
#include <vector>
#include <string>
class DataItem;
class LensContext;
class ParameterSet;
#include "Functor.h"

class InAttrDefaultFunctor: public Functor
{
   public:
      InAttrDefaultFunctor();
      InAttrDefaultFunctor(const InAttrDefaultFunctor&);
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~InAttrDefaultFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:
      std::auto_ptr<ParameterSet> _pset;
      std::string _nodeModelName;

};
#endif
