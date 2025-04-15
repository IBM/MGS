// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_execute_H
#define C_execute_H
#include "Mdl.h"

#include "C_argumentToMemberMapper.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_returnType;
class DataType;

class C_execute : public C_argumentToMemberMapper {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      virtual std::string getType() const;
      C_execute(C_returnType* returnType, bool ellipsisIncluded = false);
      C_execute(C_returnType* returnType, C_generalList* argumentList
		, bool ellipsisIncluded = false);
      C_execute(const C_execute& rv);
      virtual void duplicate(std::unique_ptr<C_execute>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      void releaseDataType(std::unique_ptr<DataType>&& dt);
      virtual ~C_execute();

   private:
      C_returnType* _returnType;
      DataType* _dataType;
};


#endif // C_execute_H
