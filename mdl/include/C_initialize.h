// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_initialize_H
#define C_initialize_H
#include "Mdl.h"

#include "C_argumentToMemberMapper.h"
#include <memory>
#include <string>

class MdlContext;
class ToolBase;
class C_generalList;

class C_initialize : public C_argumentToMemberMapper {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      virtual std::string getType() const;
      C_initialize(bool ellipsisIncluded = false);
      C_initialize(C_generalList* argumentList, bool ellipsisIncluded = false);
      C_initialize(const C_initialize& rv);
      virtual void duplicate(std::unique_ptr<C_initialize>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_initialize();
};


#endif // C_initialize_H
