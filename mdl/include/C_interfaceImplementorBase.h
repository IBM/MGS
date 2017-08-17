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
      virtual void execute(MdlContext* context);
      C_interfaceImplementorBase();
      C_interfaceImplementorBase(const std::string& name, 
				 C_interfacePointerList* ipl
				 , C_generalList* gl);
      C_interfaceImplementorBase(const C_interfaceImplementorBase& rv);
      virtual void duplicate(
	 std::auto_ptr<C_interfaceImplementorBase>& rv) const;
      virtual ~C_interfaceImplementorBase();
      void executeInterfaceImplementorBase(MdlContext* context,
					   InterfaceImplementorBase* cc) const;
   protected:
      std::string _name;
      C_interfacePointerList* _interfacePointerList;
      C_generalList* _generalList;
      

};


#endif // C_interfaceImplementorBase_H
