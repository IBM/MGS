// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "SegmentDescriptor.h"
#include "Segment.h"
#include "Branch.h"
#include "Neuron.h"
#include <cassert>

SegmentDescriptor::SegmentDescriptor()
{
	//TUAN: treatment for key-size (need to update once key-size change)
#ifdef DEBUG_ASSERT
  assert (sizeof(unsigned long long) == 8);
  assert (sizeof(SegmentID) == 8);
  assert (sizeof(double) == 8);
#endif

  _maxField[uf0] = pow2(USER_FIELD_0_BITS);
  _maxField[uf1] = pow2(USER_FIELD_1_BITS);
  _maxField[branchType] = pow2(BRANCH_TYPE_BITS);
  _maxField[branchOrder] = pow2(BRANCH_ORDER_BITS);
  _maxField[computeOrder] = pow2(COMPUTE_ORDER_BITS);
  _maxField[segmentIndex] = pow2(SEGMENT_INDEX_BITS);
  _maxField[branchIndex] = pow2(BRANCH_INDEX_BITS);
  _maxField[neuronIndex] = pow2(NEURON_INDEX_BITS);
  _maxField[flag] = pow2(FLAG_BITS);

  _fields.push_back(STRG(FIELD_0));
  _fields.push_back(STRG(FIELD_1));
  _fields.push_back(STRG(FIELD_2));
  _fields.push_back(STRG(FIELD_3));
  _fields.push_back(STRG(FIELD_4));
  _fields.push_back(STRG(FIELD_5));
  _fields.push_back(STRG(FIELD_6));
  _fields.push_back(STRG(FIELD_7));
  _fields.push_back(STRG(FIELD_8));

  _fieldMap[STRG(FIELD_0)]=SegmentKeyData(0);
  _fieldMap[STRG(FIELD_1)]=SegmentKeyData(1);
  _fieldMap[STRG(FIELD_2)]=SegmentKeyData(2);
  _fieldMap[STRG(FIELD_3)]=SegmentKeyData(3);
  _fieldMap[STRG(FIELD_4)]=SegmentKeyData(4);
  _fieldMap[STRG(FIELD_5)]=SegmentKeyData(5);
  _fieldMap[STRG(FIELD_6)]=SegmentKeyData(6);
  _fieldMap[STRG(FIELD_7)]=SegmentKeyData(7);
  _fieldMap[STRG(FIELD_8)]=SegmentKeyData(8);
  _fieldMap["USER_FIELD_0"]=uf0;
  _fieldMap["USER_FIELD_1"]=uf1;

  sid.idLong = 0;
  sid.key.userField0 = _maxField[uf0]-1;
  _mask[uf0] = sid.idLong;

  sid.idLong = 0;
  sid.key.userField1 = _maxField[uf1]-1;
  _mask[uf1] = sid.idLong;

  sid.idLong = 0;
  sid.key.branchType = _maxField[branchType]-1;
  _mask[branchType] = sid.idLong;

  sid.idLong = 0;
  sid.key.branchOrder = _maxField[branchOrder]-1;
  _mask[branchOrder] = sid.idLong;

  sid.idLong = 0;
  sid.key.computeOrder = _maxField[computeOrder]-1;
  _mask[computeOrder] = sid.idLong;

  sid.idLong = 0;
  sid.key.segmentIndex = _maxField[segmentIndex]-1;
  _mask[segmentIndex] = sid.idLong;

  sid.idLong = 0;
  sid.key.branchIndex = _maxField[branchIndex]-1;  
  _mask[branchIndex] = sid.idLong;

  sid.idLong = 0;
  sid.key.neuronIndex = _maxField[neuronIndex]-1;
  _mask[neuronIndex] = sid.idLong;

  sid.idLong = 0;
  sid.key.flag = _maxField[flag]-1;  
  _mask[flag] = sid.idLong;
}

SegmentDescriptor::SegmentDescriptor(SegmentDescriptor const & segmentDescriptor) :
  _fields(segmentDescriptor._fields),
  _fieldMap(segmentDescriptor._fieldMap)
{
  for (int i=0; i<N_FIELDS; ++i) {
    _mask[i] = segmentDescriptor._mask[i];
    _maxField[i]=segmentDescriptor._maxField[i];
  }
  sid = segmentDescriptor.sid;
}

SegmentDescriptor::SegmentDescriptor(SegmentDescriptor& segmentDescriptor) :
  _fields(segmentDescriptor._fields),
  _fieldMap(segmentDescriptor._fieldMap)
{
  for (int i=0; i<N_FIELDS; ++i) {
    _mask[i] = segmentDescriptor._mask[i];
    _maxField[i]=segmentDescriptor._maxField[i];
  }

  sid = segmentDescriptor.sid;
}

SegmentDescriptor::~SegmentDescriptor()
{
}


key_size_t SegmentDescriptor::getSegmentKey(Segment* segment)
{
  Branch* branch = segment->getBranch();
  Neuron* neuron = branch->getNeuron();

  sid.idLong = 0;

  int value;

  value=neuron->getMorphologicalType();
  assert(value<_maxField[uf0]);
  sid.key.userField0 = value;

  value=neuron->getElectrophysiologicalType();
  assert(value<_maxField[uf1]);
  sid.key.userField1 = value;

  value=branch->getBranchType();
  assert(value<_maxField[branchType]);
  sid.key.branchType = value;

  value=branch->getBranchOrder();
  assert(value<_maxField[branchOrder]);
  sid.key.branchOrder = value;

  value=segment->getComputeOrder();
  assert(value<_maxField[computeOrder]);
  sid.key.computeOrder = value;

  value=segment->getSegmentIndex();
  assert (value<_maxField[segmentIndex]);
  sid.key.segmentIndex = value;

  value=branch->getBranchIndex();
  assert(value<_maxField[branchIndex]);
  sid.key.branchIndex = value;
  
  value=neuron->getGlobalNeuronIndex();
  assert(value<_maxField[neuronIndex]);
  sid.key.neuronIndex = value;

  value=(segment->isJunctionSegment() ? 1 : 0);
  assert(value<_maxField[flag]);
  sid.key.flag = value;

  return sid.collapsed;
}

key_size_t SegmentDescriptor::flipFlag(key_size_t key)
{
  sid.collapsed=key;
  sid.key.flag = (sid.key.flag==0) ? 1 : 0;
  return sid.collapsed;
}

//modify the existing key in 'key' using the value given in 'id'
// of the field in 'field'
key_size_t SegmentDescriptor::modifySegmentKey(SegmentKeyData field, unsigned int id, key_size_t key)
{
  sid.collapsed=key;
  switch(field) {
  case uf0:
    assert(id<_maxField[uf0]);
    sid.key.userField0=id; 
    break;
  case uf1:  
    assert(id<_maxField[uf1]);
    sid.key.userField1=id; 
    break;
  case branchType:  
    assert(id<_maxField[branchType]);
    sid.key.branchType=id; 
    break;
  case branchOrder:
    assert(id<_maxField[branchOrder]);
    sid.key.branchOrder=id; 
    break;
  case computeOrder: 
    assert(id<_maxField[computeOrder]);
    sid.key.computeOrder=id; 
    break;
  case segmentIndex: 
    assert(id<_maxField[segmentIndex]);
    sid.key.segmentIndex=id; 
    break;
  case branchIndex:  
    assert(id<_maxField[branchIndex]);
    sid.key.branchIndex=id; 
    break;
  case neuronIndex:  
    assert(id<_maxField[neuronIndex]);
    sid.key.neuronIndex=id; 
    break;
  case flag: 
    assert(id<_maxField[flag]);
    sid.key.flag=id; 
    break;
  default : break;
  }
  return sid.collapsed;
}

key_size_t SegmentDescriptor::getSegmentKey(std::vector<SegmentKeyData> const & maskVector, unsigned int* ids)
{
  sid.idLong = 0;

  std::vector<SegmentKeyData>::const_iterator iter = maskVector.begin(), end = maskVector.end();
  for (int i=0; iter!=end; ++iter, ++i)
    sid.collapsed=modifySegmentKey(*iter, ids[i], sid.collapsed);
  return sid.collapsed;
}

//GOAL: the returned value called 'mask': will be used by getSegmentKey(true-key, mask)
unsigned long long SegmentDescriptor::getMask(std::vector<SegmentKeyData> const &  maskVector)
{
  unsigned long long mask = 0;
  std::vector<SegmentKeyData>::const_iterator iter = maskVector.begin(), end = maskVector.end();
  for (; iter!=end; ++iter) mask |= _mask[*iter];
  return mask;
}

//GOAL: build an abstract-key based on the true key 'segmentKey' 
//          and the mask value returned from getMask() 
key_size_t SegmentDescriptor::getSegmentKey(key_size_t segmentKey, unsigned long long mask)
{
  sid.collapsed = segmentKey;
  sid.idLong &= mask;
  return sid.collapsed;
}

unsigned int SegmentDescriptor::getValue(SegmentKeyData skd, key_size_t segmentKey)
{
  unsigned int rval=0;
  sid.collapsed = segmentKey;
  switch (skd) {

  case uf0:
    rval=sid.key.userField0;
    break;
  case uf1:  
    rval=sid.key.userField1;
    break;
  case branchType:  
    rval=sid.key.branchType;
    break;
  case branchOrder:
    rval=sid.key.branchOrder;
    break;
  case computeOrder: 
    rval=sid.key.computeOrder;
    break;
  case neuronIndex:  
    rval=sid.key.neuronIndex;
    break;
  case branchIndex:  
    rval=sid.key.branchIndex;
    break;
  case segmentIndex: 
    rval=sid.key.segmentIndex;
    break;
  case flag: 
    rval=sid.key.flag;
    break;
			
  default : break;
  }

  return rval;
}

//INPUT:
//  fieldName = one of the 9 field names
//OUTPUT:
//  return the index to the given fieldname
SegmentDescriptor::SegmentKeyData SegmentDescriptor::getSegmentKeyData(std::string fieldName)
{
  std::map<std::string, SegmentDescriptor::SegmentKeyData>::iterator miter=_fieldMap.find(fieldName);
  if (miter==_fieldMap.end()) {
    std::cerr<<"Unrecognized Segment Key field name: "<<fieldName<<std::endl;
    exit(1);
  }
  return miter->second;
}


unsigned int SegmentDescriptor::getBranchType(key_size_t segmentKey)
{
  sid.collapsed = segmentKey;
  return sid.key.branchType;
}

unsigned int SegmentDescriptor::getBranchOrder(key_size_t segmentKey)
{
  sid.collapsed = segmentKey;
  return sid.key.branchOrder;
}

unsigned int SegmentDescriptor::getComputeOrder(key_size_t segmentKey)
{
  sid.collapsed = segmentKey;
  return sid.key.computeOrder;
}

unsigned int SegmentDescriptor::getSegmentIndex(key_size_t segmentKey)
{
  sid.collapsed = segmentKey;
  return sid.key.segmentIndex;
}

unsigned int SegmentDescriptor::getBranchIndex(key_size_t segmentKey)
{
  sid.collapsed = segmentKey;
  return sid.key.branchIndex;
}

unsigned int SegmentDescriptor::getNeuronIndex (key_size_t segmentKey)
{
  sid.collapsed = segmentKey;
  return sid.key.neuronIndex;
}

bool SegmentDescriptor::getFlag(key_size_t segmentKey)
{
  sid.collapsed = segmentKey;
  return (sid.key.flag!=0);
}

unsigned int SegmentDescriptor::pow2(int n)
{
  unsigned int rval = 2;
  for (; n>1; --n) rval *= 2;
  return rval;
}

void SegmentDescriptor::printKey(key_size_t segmentKey, unsigned long long mask)
{
  sid.collapsed = segmentKey;
  SegmentKeyData skd=SegmentKeyData(0);
  SegmentID sid2;
  sid2.idLong=mask;
  int i=0;

  while(i<N_FIELDS) {
    SegmentKeyData skd=SegmentKeyData(i);

    switch (skd) {

    case uf0:
      if (sid2.key.userField0) 
	printf("%s=%u", _fields[uf0].c_str(), sid.key.userField0);
      else assert(sid.key.userField0==0);
      break;
    case uf1:  
      if (sid2.key.userField1)
	printf("%s=%u", _fields[uf1].c_str(), sid.key.userField1);
      else assert(sid.key.userField1==0);
      break;
    case branchType:  
      if (sid2.key.branchType)
	printf("%s=%u", _fields[branchType].c_str(), sid.key.branchType);
      else assert(sid.key.branchType==0);
      break;
    case branchOrder:
      if (sid2.key.branchOrder)
	printf("%s=%u", _fields[branchOrder].c_str(), sid.key.branchOrder);
      else assert(sid.key.branchOrder==0);
      break;
    case computeOrder: 
      if (sid2.key.computeOrder)
	printf("%s=%u", _fields[computeOrder].c_str(), sid.key.computeOrder);
      else assert(sid.key.computeOrder==0);
      break;
    case segmentIndex: 
      if (sid2.key.segmentIndex)
	printf("%s=%u", _fields[segmentIndex].c_str(), sid.key.segmentIndex);
      else assert(sid.key.segmentIndex==0);
      break;
    case branchIndex:
      if (sid2.key.branchIndex)
	printf("%s=%u", _fields[branchIndex].c_str(), sid.key.branchIndex);
      else assert(sid.key.branchIndex==0);
      break;
    case neuronIndex:
      if (sid2.key.neuronIndex)
	printf("%s=%u", _fields[neuronIndex].c_str(), sid.key.neuronIndex);
      else assert(sid.key.neuronIndex==0);
      break;
    case flag:
      if (sid2.key.flag)
	printf("%s=%u", _fields[flag].c_str(), sid.key.flag);
      else assert(sid.key.flag==0);
      break;
    default : break;
    }
    if (++i!=N_FIELDS) printf(" | ");
  }
}

void SegmentDescriptor::printMask(unsigned long long mask)
{
  sid.idLong=mask;
  if (sid.key.userField0) printf("%s ", _fields[uf0].c_str());
  if (sid.key.userField1) printf("%s ", _fields[uf1].c_str());
  if (sid.key.branchType) printf("%s ", _fields[branchType].c_str());
  if (sid.key.branchOrder) printf("%s ", _fields[branchOrder].c_str());
  if (sid.key.computeOrder) printf("%s ", _fields[computeOrder].c_str());
  if (sid.key.segmentIndex) printf("%s ", _fields[segmentIndex].c_str());
  if (sid.key.branchIndex) printf("%s ", _fields[branchIndex].c_str());
  if (sid.key.neuronIndex) printf("%s ", _fields[neuronIndex].c_str());
  if (sid.key.flag) printf("%s ", _fields[flag].c_str());
  printf("\n");
}

unsigned long long SegmentDescriptor::getLongKey(key_size_t key)
{
  sid.collapsed=key;
  return sid.idLong;
}

SegmentDescriptor _SegmentDescriptor;

std::string SegmentDescriptor::getFieldName(const int & fieldValue)
{ 
	return _fields[fieldValue];
}

