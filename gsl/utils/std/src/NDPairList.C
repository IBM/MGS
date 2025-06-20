// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NDPairList.h"
#include "NDPair.h"

#include <string>
#include <memory>
#include <list>
#include <sstream>

NDPairList::NDPairList()
{
}

NDPairList::NDPairList(const NDPairList& rv)
{
   copyContents(rv);
}

NDPairList& NDPairList::operator=(const NDPairList& rv)
{
   if (this == &rv) {
      return *this;
   }
   destructContents();
   copyContents(rv);
   return *this;
}

void NDPairList::duplicate(std::unique_ptr<NDPairList>&& dup) const
{
   dup.reset(new NDPairList(*this));
}


NDPairList::~NDPairList()
{
   destructContents();
}

void NDPairList::copyContents(const NDPairList& rv)
{
   std::list<NDPair*>::const_iterator it, end = rv.end();
   for (it = rv.begin(); it != end; it++) {
      _data.push_back(new NDPair(**it));
   }
}

void NDPairList::destructContents()
{
   std::list<NDPair*>::iterator it, end = _data.end();
   for (it = _data.begin(); it != end; it++) {
      delete *it;
   }     
   this->clear();
}

bool NDPairList::replace(const std::string& name, const std::string& value)
{
  bool rval=false;
  std::list<NDPair*>::iterator it, end = _data.end();
  for (it = _data.begin(); it != end; it++) {
    if ((*it)->getName()==name) {
      (*it)->setValue(value);
      rval=true;
      break;
    }
  }
  return rval;
}

bool NDPairList::replace(const std::string& name, int value)
{
  bool rval=false;
  std::list<NDPair*>::iterator it, end = _data.end();
  for (it = _data.begin(); it != end; it++) {
    if ((*it)->getName()==name) {
      (*it)->setValue(value);
      rval=true;
      break;
    }
  }
  return rval;
}

bool NDPairList::replace(const std::string& name, double value)
{
  bool rval=false;
  std::list<NDPair*>::iterator it, end = _data.end();
  for (it = _data.begin(); it != end; it++) {
    if ((*it)->getName()==name) {
      (*it)->setValue(value);
      rval=true;
      break;
    }
  }
  return rval;
}

bool NDPairList::replace(const std::string& name, std::unique_ptr<DataItem>& di)
{
  bool rval=false;
  std::list<NDPair*>::iterator it, end = _data.end();
  for (it = _data.begin(); it != end; it++) {
    if ((*it)->getName()==name) {
      (*it)->setDataItemOwnership(di);
      rval=true;
      break;
    }
  }
  return rval;
}
