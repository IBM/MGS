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

#include "NodeSetArrayDataItem.h"
#include "VolumeOdometer.h"
#include <iostream>
#include "NodeSet.h"
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* NodeSetArrayDataItem::_type = "NODE_SET_ARRAY";

// Constructors
NodeSetArrayDataItem::NodeSetArrayDataItem()
{
   _data = new std::vector<NodeSet*>;
}


NodeSetArrayDataItem::NodeSetArrayDataItem(const NodeSetArrayDataItem& DI)
{
   _data = new std::vector<NodeSet*>;
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   std::vector<NodeSet*> const &v = *(DI.getNodeSetVector());
   std::vector<NodeSet*>::const_iterator iter = v.begin();
   std::vector<NodeSet*>::const_iterator end = v.end();
   for(; iter != end; ++iter) _data->push_back(new NodeSet(**iter));
}


// Utility methods
void NodeSetArrayDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new NodeSetArrayDataItem(*this)));
}


NodeSetArrayDataItem& NodeSetArrayDataItem::assign(const NodeSetArrayDataItem& DI)
{
   // clean up old
   std::vector<NodeSet*>::const_iterator iter = _data->begin();
   std::vector<NodeSet*>::const_iterator end = _data->end();
   for(; iter != end; ++iter) delete(*iter);
   _data->clear();

   // build new
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   std::vector<NodeSet*> const &v = *(DI.getNodeSetVector());
   iter = v.begin();
   end = v.end();
   for(; iter != end; ++iter) _data->push_back(new NodeSet(**iter));
   return(*this);
}


const char* NodeSetArrayDataItem::getType() const
{
   return _type;
}


// Array methods

NodeSet* NodeSetArrayDataItem::getNodeSet(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"NodeSetArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (*_data)[offset];
}


void NodeSetArrayDataItem::setNodeSet(std::vector<int> coords, NodeSet* value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"NodeSetArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = new NodeSet(*value);
}


const std::vector<NodeSet*>* NodeSetArrayDataItem::getNodeSetVector() const
{
   return _data;
}


std::vector<NodeSet*>* NodeSetArrayDataItem::getModifiableNodeSetVector()
{
   return _data;
}


NodeSetArrayDataItem::~NodeSetArrayDataItem()
{
   std::vector<NodeSet*>::iterator iter = _data->begin();
   std::vector<NodeSet*>::iterator end = _data->end();
   for(; iter != end; ++iter) delete(*iter);
}


NodeSetArrayDataItem::NodeSetArrayDataItem(std::vector<int> const &dimensions)
:ArrayDataItem(dimensions), _data(0)
{
   int size=1;
   std::vector<int>::iterator i, end =_dimensions.end();
   for(i=_dimensions.begin();i!=end;++i) size *= *i;
   //_data->resize(size);
   _data = new std::vector<NodeSet*>();
   for(int i=0; i < size; i++) {
      _data->push_back(0);
   }
   //_data->assign(size,0);
}

NodeSetArrayDataItem::NodeSetArrayDataItem(ShallowArray<NodeSet*> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<NodeSet*>(size);  
  ShallowArray<NodeSet*>::const_iterator iter=data.begin(), end=data.end();
  std::vector<NodeSet*>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}


// Utility method
void NodeSetArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   std::unique_ptr<DataItem*> diap;
   unsigned size=1;
   std::vector<int>::iterator iter, dend =_dimensions.end();
   for(iter=_dimensions.begin();iter!=dend;++iter) size *= *iter;
   std::vector<NodeSet*> *dest = new std::vector<NodeSet*>(size);
   dest->assign(dest->size(),0);

   // find overlap between old dimensions and new
   unsigned oldsize = _dimensions.size();
   unsigned newsize = dimensions.size();
   int minNumDimensions = (oldsize> newsize)?oldsize:newsize;
   std::vector<int> minDimensions(minNumDimensions);
   std::vector<int> begin(minNumDimensions);
   for(int i = 0;i<minNumDimensions;++i) {
      begin[i] = 0;
      minDimensions[i] = (_dimensions[i]>dimensions[i])? _dimensions[i]:dimensions[i];
   }

   // Copy appropriate DataItem pointers to new vector
   VolumeOdometer odmtr(begin, minDimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() ) {
      unsigned sourceOffset = getOffset(coords);
      unsigned destOffset = getOffset(dimensions,coords);
      (*dest)[destOffset] = (*_data)[sourceOffset];
      (*_data)[sourceOffset]=0;
   }

   // get rid of any DataItems not retained
   for (unsigned i=0;i<_data->size();++i) {
      delete (*_data)[i];
   }

   // replace old vector with new
   delete _data;
   _data = dest;
   ArrayDataItem::_setDimensions(dimensions);
}
