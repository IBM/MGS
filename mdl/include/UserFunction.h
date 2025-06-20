// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef UserFunction_H
#define UserFunction_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "Class.h"

class ConnectionCCBase;

class UserFunction {

   public:
      UserFunction(const std::string& name);
      virtual void duplicate(std::unique_ptr<UserFunction>&& rv) const;
      virtual ~UserFunction();
     
      std::string getName() const {
	 return _name;
      }

      void generateInstanceMethod(Class& instance, bool pureVirtual,
				  const ConnectionCCBase& ccBase) const;

      std::string getString() const;

   protected:
      std::string _name;
};


#endif // UserFunction _H
