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

#include "C_dataType.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "DataType.h"
#include "C_typeClassifier.h"
#include "InternalException.h"
#include "SyntaxErrorException.h"
#include "NameComment.h"
#include <memory>
#include <string>

void C_dataType::execute(MdlContext* context) 
{
   if (_typeClassifier == 0) {
      throw InternalException("_typeClassifier is 0 in C_dataType::execute");
   }
   _typeClassifier->execute(context);
   std::auto_ptr<DataType> dt;
   _typeClassifier->releaseDataType(dt);
   dt->setDerived(_derived);      
   _nameCommentList->execute(context);
   const std::vector<NameComment>& ncVec = 
      _nameCommentList->getNameCommentVec();
   _dataTypeVec = new std::vector<DataType*>;
   std::vector<NameComment>::const_iterator it, end = ncVec.end();
   std::auto_ptr<DataType> dtToIns;
   for (it = ncVec.begin(); it != end; ++it) {
      dt->duplicate(dtToIns);
      dtToIns->setName(it->getName());
      dtToIns->setComment(it->getComment());
      if (it->getBlockSize() < 0) {
	 throw SyntaxErrorException("Block size is negative."); 
      }
      if (it->getIncrementSize() < 0) {
	 throw SyntaxErrorException("Increment size is negative."); 
      }
      if ((it->getBlockSize() > 0) || (it->getIncrementSize() > 0)) {
	 if (!dtToIns->isArray()) {
	    throw SyntaxErrorException(
	       "Trying to set block size or increment size on a non-array data type."); 
	 }
	 if ((it->getIncrementSize() > 0) && (it->getBlockSize() == 0)) {
	    throw SyntaxErrorException("BlockSize can not be zero."); 
	 }
	 dtToIns->setArrayCharacteristics(
	    it->getBlockSize(), it->getIncrementSize());
      }

      _dataTypeVec->push_back(dtToIns.release());
   }
}

void C_dataType::addToList(C_generalList* gl) 
{
   std::auto_ptr<DataType> dt;
   std::vector<DataType*>::iterator it, end = _dataTypeVec->end();
   for (it = _dataTypeVec->begin(); it != end; ++it) {
      dt.reset(*it);
      if (_optional) {
	 gl->addOptionalDataType(dt);
      } else {
	 gl->addDataType(dt);
      }
   }
   delete _dataTypeVec;
   _dataTypeVec = 0;
}

C_dataType::C_dataType() 
   : C_general(), _derived(false), _optional(false), _typeClassifier(0), 
     _nameCommentList(0), _dataTypeVec(0) 
{

}

C_dataType::C_dataType(C_typeClassifier* tc, C_nameCommentList* ncl, 
		       bool derived, bool optional) 
   : C_general(), _derived(derived), _optional(optional), _typeClassifier(tc), 
     _nameCommentList(ncl), _dataTypeVec(0) 
{

}

C_dataType::C_dataType(const C_dataType& rv) 
   : C_general(rv), _derived(rv._derived), _optional(rv._optional), 
     _typeClassifier(0), _nameCommentList(0), _dataTypeVec(0) 
{
   if (rv._typeClassifier) {
      std::auto_ptr<C_typeClassifier> dup;
      rv._typeClassifier->duplicate(dup);
      _typeClassifier = dup.release();
   }
   if (rv._nameCommentList) {
      std::auto_ptr<C_nameCommentList> dup;
      rv._nameCommentList->duplicate(dup);
      _nameCommentList = dup.release();
   }
   deepCopyVector(rv);
}

void C_dataType::duplicate(std::auto_ptr<C_dataType>& rv) const
{
   rv.reset(new C_dataType(*this));
}

void C_dataType::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_dataType(*this));
}

void C_dataType::releaseDataTypeVec(
   std::auto_ptr<std::vector<DataType*> >& dtv) 
{
   dtv.reset(_dataTypeVec);
   _dataTypeVec = 0;
}

C_dataType::~C_dataType() 
{
   delete _typeClassifier;
   delete _nameCommentList;
   deleteVector();
}

void C_dataType::deleteVector() 
{
   if (_dataTypeVec) {
      std::vector<DataType*>::iterator end = _dataTypeVec->end();
      std::vector<DataType*>::iterator it;
      for (it = _dataTypeVec->begin(); it != end; it++) {
	 delete *it;
      }
      delete _dataTypeVec;
      _dataTypeVec = 0;
   }
}

void C_dataType::deepCopyVector(const C_dataType& rv) 
{
   if (rv._dataTypeVec) {
      _dataTypeVec = new std::vector<DataType*>();
      std::vector<DataType*>::const_iterator end = rv._dataTypeVec->end();
      std::vector<DataType*>::const_iterator it;
      std::auto_ptr<DataType> dup;   
      for (it = rv._dataTypeVec->begin(); it != end; it++) {
	 (*it)->duplicate(dup);
	 _dataTypeVec->push_back(dup.release());
      }
   }
}
