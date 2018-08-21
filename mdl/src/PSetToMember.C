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

#include "PSetToMember.h"
#include "StructType.h"
#include "DataType.h"
#include "ArrayType.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "NotFoundException.h"
#include "DuplicateException.h"
#include "MemberContainer.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <map>
#include <iostream>

PSetToMember::PSetToMember(StructType* pset) 
{
   setPSet(pset);
}

void PSetToMember::addMapping(const std::string& name, 
			      std::auto_ptr<DataType>& data) 
{
   iterator it = find(name);
   if (it != _mappings.end()) {
      std::ostringstream stream;
      stream << name << " is already in the container.";
      throw DuplicateException(stream.str()); 
   } 
   if (_pset == 0) {
      throw InternalException(
	 "_pset is 0 in PSetToMember::addMapping");
   }
   DataType* type;
   try {
      type = _pset->_members.getMember(name);
   } catch (NotFoundException& e) {
      std::string mes = "Member " + name + " is not found on PSet " 
	 + _pset->getName(); 
      e.setError(mes);
      throw;
   }
   checkAndExtraWork(name, data.get(), type);
   elemType elem;
   elem.first = name;
   elem.second = data.release();
   _mappings.push_back(elem);
}

PSetToMember::~PSetToMember() 
{
   destructOwnedHeap();
}


bool PSetToMember::existsInMappings(const std::string& token) const
{
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      if (it->first == token) {
	 return true;
      }
   }
   return false;
}

PSetToMember::iterator PSetToMember::find(
   const std::string& token)
{
   PSetToMember::iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      if (it->first == token) {
	 break;
      }
   }
   return it;
}

std::string PSetToMember::getPSetToMemberString() const
{
   std::string direction = " >> ";
   std::string tab = "\t\t";
   std::ostringstream os;
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
	 os  << tab << "PSet." << it->first << direction;
	 if (it->second->isShared()) {
	    os << "Shared.";
	 }
	 os << it->second->getName() << ";\n"; 
   }   
   return os.str();
}

void PSetToMember::duplicate(std::auto_ptr<PSetToMember>& rv) const
{
   rv.reset(new PSetToMember(*this));
}

void PSetToMember::checkAndExtraWork(const std::string& name,
   DataType* member, const DataType* pset) {
   
   const DataType* realMember;

   const std::vector<std::string>& subAttributePath = 
      member->getSubAttributePath();

   if (subAttributePath.size() > 0) {    
      if (member->isArray()) {
	 ArrayType* array = static_cast<ArrayType*>(member);
	 realMember = array->getType();
	 _mappingType.push_back(ONETOMANY);
      } else {
	 realMember = member;
	 _mappingType.push_back(ONETOONE);
      }

      const StructType* nextStruct;
      const DataType* nextMember;
      nextStruct = dynamic_cast<const StructType*>(realMember);

      if (nextStruct == 0) {
	 std::ostringstream os;
 	 os << realMember->getName()
 	    << " is not a struct type"; 
 	 throw GeneralException(os.str());
      }

      if (nextStruct->isPointer()) {
	 std::ostringstream os;
 	 os << nextStruct->getName()
 	    << " can not be a pointer"; 
 	 throw GeneralException(os.str());
      }
      
      std::vector<std::string>::const_iterator it, next, end 
	 = subAttributePath.end();
      for (it = subAttributePath.begin(); it != end; ++it) { 
	 try { 
	    nextMember = 
	       nextStruct->_members.getMember(*it);
	 } catch (NotFoundException& e) {
	    std::ostringstream os;
	    os << *it << " does not exist in struct " 
	       << nextStruct->getTypeName(); 
	    throw GeneralException(os.str());
	 }
	 
	 next = it + 1;
	 if (next == end) {
	    if (nextMember->getTypeString() != pset->getTypeString()) {
	       std::ostringstream os;
	       os << " ParameterSets member " 
		  <<  name 
		  << " is of type " << pset->getTypeString() << " not " 
		  << nextMember->getTypeString() << " ( " 
		  << nextMember->getName() << "'s type)";
	       throw GeneralException(os.str());
	    }    
	 } else {
	    nextStruct = dynamic_cast<const StructType*>(nextMember);
	    
	    if (nextStruct == 0) {
	       std::ostringstream os;
	       os << nextMember->getName()
		  << " is not a struct type"; 
	       throw GeneralException(os.str());
	    }	    

	    if (nextStruct->isPointer()) {
	       std::ostringstream os;
	       os << nextStruct->getName()
		  << " can not be a pointer"; 
	       throw GeneralException(os.str());
	    }
	 }
      }
   } else {
      if (member->isArray() && !pset->isArray()) {
	 ArrayType* array = static_cast<ArrayType*>(member);
	 realMember = array->getType();
	 _mappingType.push_back(ONETOMANY);
      } else {
	 realMember = member;
	 _mappingType.push_back(ONETOONE);
      }

      if (realMember->getDescriptor() != pset->getDescriptor()) {
	 std::ostringstream os;
	 os << " pset " << _pset->getName() << "'s member " <<  name 
	    << " is of type " << pset->getDescriptor() << " not " 
	    << realMember->getDescriptor();
	 throw GeneralException(os.str());
      }
   }
}

std::string PSetToMember::getPSetToMemberCode(
   const std::string& tab, std::set<std::string>& requiredIncreases) const
{
   std::string psetName = INATTRPSETNAME;   
   std::string memberName;

   std::ostringstream os;
   std::vector<MappingType>::const_iterator mIt = _mappingType.begin();
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it, ++mIt) {
      const std::vector<std::string>& subAttributePath = 
	 it->second->getSubAttributePath();
      std::string path = "";
      
      std::vector<std::string>::const_iterator sit, send 
	 = subAttributePath.end();
      for (sit = subAttributePath.begin(); sit != send; ++sit) { 
	 path += "." + *sit;
      }

      memberName = "";
      if (it->second->isShared()) {
	 memberName = "getNonConstSharedMembers().";
      }
      memberName += it->second->getName();
      if (*mIt == ONETOONE) {
	 os << tab << memberName << " = " << psetName << "->" 
	    << PREFIX << "get_" << _pset->getName() << "_" << it->first 
	    << "();\n";
      } else { // ONETOMANY
	 if (path == "") {
	    os << tab << memberName << ".insert(" << psetName << "->" 
	       << it->first << ");\n";
	 } else {
  	    std::string sizeName = PREFIX + memberName + "Size";
	    requiredIncreases.insert(memberName);
	    os << tab << memberName << "[" << sizeName << "]" 
	       << path << " = " << psetName << "->" 
	       << it->first << ";\n";	    
	 }
      }
   }
   return os.str();
}

PSetToMember::PSetToMember(const PSetToMember& rv)
   : _pset(rv._pset), _mappingType(rv._mappingType)
{
   copyOwnedHeap(rv);
}

PSetToMember& PSetToMember::operator=(const PSetToMember& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _pset = rv._pset; 
      _mappingType = rv._mappingType;
   }
   return *this;
}

void PSetToMember::destructOwnedHeap()
{
   std::vector<elemType>::iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      delete it->second;
   }
   _mappings.clear();
}

void PSetToMember::copyOwnedHeap(const PSetToMember& rv)
{
   if (rv._mappings.size() > 0) {
      std::vector<elemType>::const_iterator it, end = rv._mappings.end();
      for (it = _mappings.begin(); it != end; ++it) {
	 elemType elem;
	 elem.first = it->first;
	 std::auto_ptr<DataType> dup;
	 it->second->duplicate(dup);
	 elem.second = dup.release();
	 _mappings.push_back(elem);
      }      
   }
}
