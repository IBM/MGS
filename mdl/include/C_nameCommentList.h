// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_nameCommentList_H
#define C_nameCommentList_H
#include "Mdl.h"

#include "C_production.h"
#include "NameComment.h"
#include "C_nameComment.h"
#include <memory>
#include <vector>

class MdlContext;
class C_nameComment;
class DataType;

class C_nameCommentList : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_nameCommentList();
      C_nameCommentList(C_nameComment* dt);
      C_nameCommentList(C_nameCommentList* dtl, C_nameComment* dt);
      C_nameCommentList(const C_nameCommentList& rv);
      virtual void duplicate(std::unique_ptr<C_nameCommentList>&& rv) const;
      virtual ~C_nameCommentList();
      const std::vector<NameComment>& getNameCommentVec() {
	 return _nameCommentVec;
      }

   private:
      C_nameComment* _nameComment;
      C_nameCommentList* _nameCommentList;
      std::vector<NameComment> _nameCommentVec;
};


#endif // C_nameCommentList_H
