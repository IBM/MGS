// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_interfaceImplementorBase_H
#define C_interfaceImplementorBase_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class InterfaceImplementorBase;

class C_interfaceImplementorBase : public C_production {
   public:
      using C_production::duplicate;  // Make base class method visible
      virtual void execute(MdlContext* context);
      C_interfaceImplementorBase();
      C_interfaceImplementorBase(const std::string& name, 
				 C_interfacePointerList* ipl
				 , C_generalList* gl);
      C_interfaceImplementorBase(const C_interfaceImplementorBase& rv);
      virtual void duplicate(
	 std::unique_ptr<C_interfaceImplementorBase>&& rv) const;
      virtual ~C_interfaceImplementorBase();
      void executeInterfaceImplementorBase(MdlContext* context,
					   InterfaceImplementorBase* cc) const;
   protected:
      std::string _name;
      C_interfacePointerList* _interfacePointerList;
      C_generalList* _generalList;
      

};


#endif // C_interfaceImplementorBase_H
