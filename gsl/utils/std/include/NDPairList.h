// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef NDPAIRLIST_H
#define NDPAIRLIST_H
#include "Copyright.h"

#include "NDPair.h"

#include <string>
#include <memory>
#include <list>

class NDPairList
{
   public:
      typedef std::list<NDPair*>::iterator iterator;
      typedef std::list<NDPair*>::const_iterator const_iterator;
      typedef std::list<NDPair*>::reverse_iterator reverse_iterator;
      typedef std::list<NDPair*>::const_reverse_iterator const_reverse_iterator;
      NDPairList();
      NDPairList(const NDPairList& rv);
      NDPairList& operator=(const NDPairList& rv);
      virtual void duplicate(std::auto_ptr<NDPairList>& dup) const;
      virtual ~NDPairList();

      unsigned size() {
	return _data.size();
      }

      // Wrapper funcs for std::list -Begin-
      iterator begin() {
	 return _data.begin();
      }
      const_iterator begin() const {
	 return _data.begin();
      }
      iterator end() {
	 return _data.end();
      }
      const_iterator end() const {
	 return _data.end();
      }
      NDPair* back() const {
	 return _data.back();
      }
      reverse_iterator rbegin() {
	 return _data.rbegin();
      }
      const_reverse_iterator rbegin() const {
	 return _data.rbegin();
      }
      reverse_iterator rend() {
	 return _data.rend();
      }
      const_reverse_iterator rend() const {
	 return _data.rend();
      }
      void push_back(NDPair* data) {
	 _data.push_back(data);
      }
      void push_front(NDPair* data) {
	 _data.push_front(data);
      }
      void clear() {
	 _data.clear();
      }
      void erase (iterator p) {
	 _data.erase(p);
      }
      void splice (iterator p, NDPairList& x) {
	 _data.splice(p, x.getData());
      }
      void splice (iterator p, NDPairList& x, iterator i) {
	 _data.splice(p, x.getData(), i);
      }
      void splice (iterator p, NDPairList& x, iterator first, iterator last) {
	 _data.splice(p, x.getData(), first, last);
      }
      // Wrapper funcs for std::list -End-

      bool replace(const std::string&, const std::string&);
      bool replace(const std::string&, int);
      bool replace(const std::string&, double);
      bool replace(const std::string&, std::auto_ptr<DataItem>&);
      std::list<NDPair*>& getData() {
	 return _data;
      }

   private:
      void copyContents(const NDPairList& rv);
      void destructContents();
      std::list<NDPair*> _data;
};
#endif
