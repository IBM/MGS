// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_argumentToMemberMapper_H
#define C_argumentToMemberMapper_H
#include "Mdl.h"

#include "MemberContainer.h"
#include "DataType.h"
#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;

class C_argumentToMemberMapper : public C_general {

   public:
      virtual void execute(MdlContext* context);
      virtual void executeMapper(MdlContext* context,
				 MemberContainer<DataType>& members,
				 bool& ellipsisIncluded);
      virtual void addToList(C_generalList* gl)=0;
      virtual std::string getType() const =0;
      C_argumentToMemberMapper(bool ellipsisIncluded = false);
      C_argumentToMemberMapper(C_generalList* argumentList,
			       bool ellipsisIncluded = false);
      C_argumentToMemberMapper(const C_argumentToMemberMapper& rv);
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const =0;
      virtual ~C_argumentToMemberMapper();
      
   private:
      C_generalList* _argumentList;
      bool _ellipsisIncluded;
};


#endif // C_argumentToMemberMapper_H
