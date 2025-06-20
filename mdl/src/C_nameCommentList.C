// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_nameCommentList.h"
#include "MdlContext.h"
#include "C_nameComment.h"
#include "DataType.h"
#include "InternalException.h"
#include <memory>
#include <vector>

void C_nameCommentList::execute(MdlContext* context) 
{
   _nameCommentVec.clear();
   if (_nameComment == 0) {
      throw InternalException(
	 "_nameComment is 0 in C_nameCommentList::execute");
   }
   _nameComment->execute(context);
   if (_nameCommentList) {
      _nameCommentList->execute(context);
      _nameCommentVec = _nameCommentList->getNameCommentVec();      
   }
   NameComment nc(_nameComment->getName(), _nameComment->getComment(), 
		  _nameComment->getBlockSize(), 
		  _nameComment->getIncrementSize());
   _nameCommentVec.push_back(nc);
}

C_nameCommentList::C_nameCommentList() 
   : C_production(), _nameComment(0), _nameCommentList(0), _nameCommentVec(0) 
{

}

C_nameCommentList::C_nameCommentList(C_nameComment* dt) 
   : C_production(), _nameComment(dt), _nameCommentList(0), _nameCommentVec(0) 
{

}

C_nameCommentList::C_nameCommentList(
   C_nameCommentList* dtl, C_nameComment* dt) 
   : C_production(), _nameComment(dt), _nameCommentList(dtl), 
     _nameCommentVec(0) 
{

}

C_nameCommentList::C_nameCommentList(const C_nameCommentList& rv) 
   : C_production(rv), _nameComment(0), _nameCommentList(0), 
     _nameCommentVec(0) 
{
   if (rv._nameComment) {
      std::unique_ptr<C_nameComment> dup;
      rv._nameComment->duplicate(std::move(dup));
      _nameComment = dup.release();
   }
   if (rv._nameCommentList) {
      std::unique_ptr<C_nameCommentList> dup;
      rv._nameCommentList->duplicate(std::move(dup));
      _nameCommentList = dup.release();
   }
}

void C_nameCommentList::duplicate(std::unique_ptr<C_nameCommentList>&& rv) const
{
   rv.reset(new C_nameCommentList(*this));
}

C_nameCommentList::~C_nameCommentList() 
{
   delete _nameComment;
   delete _nameCommentList;
}
