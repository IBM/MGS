// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

   public:
      virtual void execute(MdlContext* context);
      C_nameCommentList();
      C_nameCommentList(C_nameComment* dt);
      C_nameCommentList(C_nameCommentList* dtl, C_nameComment* dt);
      C_nameCommentList(const C_nameCommentList& rv);
      virtual void duplicate(std::auto_ptr<C_nameCommentList>& rv) const;
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
