// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_userFunctionCall_H
#define C_userFunctionCall_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;

class C_userFunctionCall : public C_general {
   protected:
      using C_general::duplicate;
   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_userFunctionCall();
      C_userFunctionCall(const std::string& userFunctionCall); 
      C_userFunctionCall(const C_userFunctionCall& rv);
      virtual void duplicate(std::unique_ptr<C_userFunctionCall>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_userFunctionCall();
      
   private:
      std::string _userFunctionCall;
};


#endif // C_userFunctionCall_H
