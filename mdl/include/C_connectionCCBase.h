// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_connectionCCBase_H
#define C_connectionCCBase_H
#include "Mdl.h"

#include "C_compCategoryBase.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class ConnectionCCBase;

class C_connectionCCBase : public C_compCategoryBase {
   public:
      using C_compCategoryBase::duplicate;
      virtual void execute(MdlContext* context);
      C_connectionCCBase();
      C_connectionCCBase(const std::string& name, C_interfacePointerList* ipl
			 , C_generalList* gl);
      C_connectionCCBase(const C_connectionCCBase& rv);
      virtual void duplicate(std::unique_ptr<C_connectionCCBase>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_compCategoryBase>&& rv) const;
      virtual ~C_connectionCCBase();
      void executeConnectionCCBase(MdlContext* context,
				   ConnectionCCBase* cc) const;

};


#endif // C_connectionCCBase_H
