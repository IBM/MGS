// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_compCategoryBase_H
#define C_compCategoryBase_H
#include "Mdl.h"

#include "C_interfaceImplementorBase.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class CompCategoryBase;

class C_compCategoryBase : public C_interfaceImplementorBase {
   public:
      using C_interfaceImplementorBase::duplicate;  // Make base class method visible
      virtual void execute(MdlContext* context);
      C_compCategoryBase();
      C_compCategoryBase(const std::string& name, C_interfacePointerList* ipl
			 , C_generalList* gl);
      virtual void duplicate(std::unique_ptr<C_compCategoryBase>&& rv) const;
      virtual ~C_compCategoryBase();
      void executeCompCategoryBase(MdlContext* context,
				   CompCategoryBase* cc) const;
};


#endif // C_compCategoryBase_H
