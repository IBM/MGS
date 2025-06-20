// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef UserFunctionCall_H
#define UserFunctionCall_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "Class.h"

class UserFunctionCall {

   public:
      UserFunctionCall(const std::string& name);
      virtual void duplicate(std::unique_ptr<UserFunctionCall>&& rv) const;
      virtual ~UserFunctionCall();
     
      std::string getName() const {
	 return _name;
      }

   protected:
      std::string _name;
};


#endif // UserFunctionCall _H
