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

#ifndef C_functor_H
#define C_functor_H
#include "Mdl.h"

#include "C_toolBase.h"
#include <memory>
#include <string>

class MdlContext;

class C_functor : public C_toolBase {

   public:
      virtual void execute(MdlContext* context);
      C_functor();
      C_functor(const std::string& name, C_generalList* gl,
		std::string category = "FUNCTOR");
      C_functor(const C_functor& rv);
      virtual void duplicate(std::auto_ptr<C_functor>& rv) const;
      virtual ~C_functor();
      void setFrameWorkElement(bool val = true) {
	 _frameWorkElement = val;
      }
   private:
      std::string _category;
      bool _frameWorkElement;
};


#endif // C_functor_H
