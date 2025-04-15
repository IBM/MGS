// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_functor_H
#define C_functor_H
#include "Mdl.h"

#include "C_toolBase.h"
#include <memory>
#include <string>

class MdlContext;

class C_functor : public C_toolBase {
   using C_toolBase::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_functor();
      C_functor(const std::string& name, C_generalList* gl,
		std::string category = "FUNCTOR");
      C_functor(const C_functor& rv);
      virtual void duplicate(std::unique_ptr<C_functor>&& rv) const;
      virtual ~C_functor();
      void setFrameWorkElement(bool val = true) {
	 _frameWorkElement = val;
      }
   private:
      std::string _category;
      bool _frameWorkElement;
};


#endif // C_functor_H
