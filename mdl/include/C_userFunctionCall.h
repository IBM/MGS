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

#ifndef C_userFunctionCall_H
#define C_userFunctionCall_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;

class C_userFunctionCall : public C_general {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_userFunctionCall();
      C_userFunctionCall(const std::string& userFunctionCall); 
      C_userFunctionCall(const C_userFunctionCall& rv);
      virtual void duplicate(std::auto_ptr<C_userFunctionCall>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_userFunctionCall();
      
   private:
      std::string _userFunctionCall;
};


#endif // C_userFunctionCall_H
