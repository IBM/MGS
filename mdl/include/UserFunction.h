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
      virtual void duplicate(std::auto_ptr<UserFunction>& rv) const;
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
