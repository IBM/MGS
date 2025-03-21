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
