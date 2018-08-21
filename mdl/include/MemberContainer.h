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

#ifndef MemberContainer_H
#define MemberContainer_H
#include "Mdl.h"

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <memory>

#include "NotFoundException.h"
#include "DuplicateException.h"

template<class T>
class MemberContainer {
   public:
      typedef typename std::pair<std::string, T*> type;
      typedef typename std::vector<type >::const_iterator const_iterator;
      typedef typename std::vector<type >::iterator iterator;

      MemberContainer() {
      }
      MemberContainer(const MemberContainer<T>& rv) {
	 copyOwnedHeap(rv);
      }
      MemberContainer<T>& operator=(const MemberContainer<T>& rv) {
	 if (this == &rv) {
	    return *this;
	 }
	 destructOwnedHeap();
	 copyOwnedHeap(rv);
	 return *this;
      }
      void duplicate(std::auto_ptr<MemberContainer<T> >& rv) const {
	 rv.reset(new MemberContainer<T>(*this));
      }
      inline bool containsMember(const std::string& name) {
	 const_iterator it, end = _members.end();
	 for (it = _members.begin(); it != end; it++) {
	    if (it->first == name) {
	       return true;
	    }
	 }
	 return false;
      }
      inline T* getMember(const std::string& name) {
	 const_iterator it, end = _members.end();
	 for (it = _members.begin(); it != end; it++) {
	    if (it->first == name) {
	       return it->second;
	    }
	 }
	 std::ostringstream stream;
	 stream << name << " is not found.";
	 throw NotFoundException(stream.str()); 
	 return 0; // should not reach this point
      }
      inline const T* getMember(const std::string& name) const {
	 const_iterator it, end = _members.end();
	 for (it = _members.begin(); it != end; it++) {
	    if (it->first == name) {
	       return it->second;
	    }
	 }
	 std::ostringstream stream;
	 stream << name << " is not found.";
	 throw NotFoundException(stream.str()); 
	 return 0; // should not reach this point
      }
      template<class T2> inline T2* getMember(const std::string& name) {
	 T* found = getMember(name);
	 T2* retVal = dynamic_cast<T2*>(found);
	 if (retVal == 0) {
	    std::ostringstream stream;
	    stream << name << " is found, but has a different type.";
	    throw NotFoundException(stream.str());	    
	 }
	 return retVal;
      };
      inline T* addMember(const std::string& name, std::auto_ptr<T>& member) {
	 const_iterator it, end = _members.end();
	 for (it = _members.begin(); it != end; it++) {
	    if (it->first == name) {
	       std::ostringstream stream;
	       stream << name << " is already in the container.";
	       throw DuplicateException(stream.str()); 
	    }
	 }

	 _members.push_back(type(name, member.release()));
	 return (_members.rbegin())->second;
      } 
      inline T* addMemberToFront(const std::string& name, 
				 std::auto_ptr<T>& member) {
	 const_iterator it, end = _members.end();
	 for (it = _members.begin(); it != end; it++) {
	    if (it->first == name) {
	       std::ostringstream stream;
	       stream << name << " is already in the container.";
	       throw DuplicateException(stream.str()); 
	    }
	 }
	 _members.insert(_members.begin(), type(name, member.release()));
	 return (_members.begin())->second;
      } 
      inline bool exists(const std::string& name) const {
	 const_iterator it, end = _members.end();
	 for (it = _members.begin(); it != end; it++) {
	    if (it->first == name) {
	       return true;
	    }
	 }
	 return false;
      }
      inline const_iterator begin() const {
	 return _members.begin();
      }
      inline const_iterator end() const {
	 return _members.end();
      }
      inline iterator begin() {
	 return _members.begin();
      }
      inline iterator end() {
	 return _members.end();
      }
      inline int size() const {
	 return _members.size();
      }
      ~MemberContainer() {
	 destructOwnedHeap();
      }

   private:
      void copyOwnedHeap(const MemberContainer<T>& rv) {
	 const_iterator it, end = rv._members.end();
	 for (it = rv._members.begin(); it != end; it++) {
	    std::auto_ptr<T> dup;
	    it->second->duplicate(dup);
	    _members.push_back(type(it->first, dup.release()));
	 }
      }
      void destructOwnedHeap() {
	 iterator end = _members.end();
	 for (iterator it = _members.begin(); it != end; it++) {
	    delete it->second;
	 }
	 _members.clear();
      }      
      
      std::vector<type > _members;
};

#endif // Generatable_H
