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

#ifndef C_constant_H
#define C_constant_H
#include "Mdl.h"

#include "C_interfaceImplementorBase.h"
#include <memory>

class MdlContext;

class C_constant : public C_interfaceImplementorBase {
   using C_interfaceImplementorBase::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_constant();
      C_constant(const std::string& name, C_interfacePointerList* ipl,
		 C_generalList* gl);
      C_constant(const C_constant& rv);
      virtual void duplicate(std::unique_ptr<C_constant>&& rv) const;
      virtual ~C_constant();
};


#endif // C_constant_H
