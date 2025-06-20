// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Touch.h"
#include "Utilities.h"
#include <cassert>
#include <iostream>
#include <float.h>
#include <cfloat>
#include <algorithm>

#include "Branch.h"

MPI_Datatype* Touch::_typeTouch = 0;

SegmentDescriptor Touch::_segmentDescriptor;

Touch::Touch() {
  for (int i = 0; i < N_TOUCH_DATA; ++i) {
    _touchData[i] = 0;
  }

#ifndef LTWT_TOUCH
  for (int i = 0; i < 4; ++i) {
    _endTouch[i] = 0;
  }
  _remains = true;
#endif
}

Touch::Touch(Touch const& t) {
  for (int i = 0; i < N_TOUCH_DATA; ++i) {
    _touchData[i] = t._touchData[i];
  }
#ifndef LTWT_TOUCH
  for (int i = 0; i < 4; ++i) {
    _endTouch[i] = t._endTouch[i];
  }
  _remains = t._remains;
#endif
}

MPI_Datatype* Touch::getTypeTouch() {
  if (_typeTouch == 0) {
    Touch t;
    _typeTouch = new MPI_Datatype;
#ifndef LTWT_TOUCH
    // Create the base type without bounds
    Datatype baseTouchType(3, &t);
    baseTouchType.set(0, MPI_DOUBLE, N_TOUCH_DATA, t._touchData);
    baseTouchType.set(1, MPI_SHORT, 4, t._endTouch);
    baseTouchType.set(2, MPI_CHAR, 1, t._remains);
    MPI_Datatype baseTouchTypeMPI = baseTouchType.commit();

    // Then use MPI_Type_create_resized
    MPI_Datatype finalTouchType;
    MPI_Type_create_resized(baseTouchTypeMPI, 0, sizeof(Touch), &finalTouchType);
    MPI_Type_commit(&finalTouchType);
    *_typeTouch = finalTouchType;
#else
    // Create the base type without bounds
    Datatype baseTouchType(1, &t);
    baseTouchType.set(0, MPI_DOUBLE, N_TOUCH_DATA, t._touchData);
    MPI_Datatype baseTouchTypeMPI = baseTouchType.commit();

    // Then use MPI_Type_create_resized
    MPI_Datatype finalTouchType;
    MPI_Type_create_resized(baseTouchTypeMPI, 0, sizeof(Touch), &finalTouchType);
    MPI_Type_commit(&finalTouchType);
    *_typeTouch = finalTouchType;
#endif
  }
  return _typeTouch;
}

void Touch::readFromFile(FILE* dataFile) {
  size_t s = fread(_touchData, sizeof(double), N_TOUCH_DATA, dataFile);
#ifndef LTWT_TOUCH
  fread(_endTouch, sizeof(short), 4, dataFile);
  fread(&_remains, sizeof(bool), 1, dataFile);
#endif
}

void Touch::writeToFile(FILE* dataFile) {
  fwrite(_touchData, sizeof(double), N_TOUCH_DATA, dataFile);
#ifndef LTWT_TOUCH
  fwrite(_endTouch, sizeof(short), 4, dataFile);
  fwrite(&_remains, sizeof(bool), 1, dataFile);
#endif
}

void Touch::printTouch() {
  std::cerr << _segmentDescriptor.getNeuronIndex(_touchData[0]) << " "
            << _segmentDescriptor.getBranchIndex(_touchData[0]) << " "
            << _segmentDescriptor.getSegmentIndex(_touchData[0]) << " "
            << _segmentDescriptor.getNeuronIndex(_touchData[1]) << " "
            << _segmentDescriptor.getBranchIndex(_touchData[1]) << " "
            << _segmentDescriptor.getSegmentIndex(_touchData[1]) << " "
            << _touchData[2] << " " << _touchData[3] << " "
#ifndef LTWT_TOUCH
            << _touchData[4] << " " << _endTouch[0] << _endTouch[1]
            << _endTouch[2] << _endTouch[3] << " " << _remains
#endif
      ;
}

key_size_t Touch::getPartner(key_size_t key) {
  key_size_t rval = 0;
  if (key == getKey1())
    rval = getKey2();
  else if (key == getKey2())
    rval = getKey1();
  return rval;
}

//GOAL: return 'prop' of the given key
// NOTE: Go to TouchDetector to see how 'prop' is calculated
double Touch::getProp(key_size_t key) {
  double rval = DBL_MAX;
  if (key == getKey1())
    rval = getProp1();
  else if (key == getKey2())
    rval = getProp2();
  return rval;
}

//GOAL: 
// A touch has 2 capsule
//    check if one of the two capsules is the spineneck
//    and return the key of that capsule
// TODO: change the way we determine in the coming future
//       where the spine head+neck is not passed in via tissue file
//       but automatically generated
bool Touch::hasSpineNeck(key_size_t& key)//obsolete
{
  //TUAN TODO POTENTIAL BUG
  //right now, for single neuron scenario, 
  // if one-side MTYPE > 0 and the other-side MTYPE=0
  // indicate a spine-denshaft touch
  //     
  bool rval=false;

  key_size_t tkey1 = getKey1();
  //NOTE: uf0 = MTYPE
  unsigned int mtype1 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey1);
  unsigned int brchtype1 = _segmentDescriptor.getBranchType(tkey1);
  key_size_t tkey2 = getKey2();
  unsigned int mtype2 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey2);
  unsigned int brchtype2 = _segmentDescriptor.getBranchType(tkey2);
  if (mtype1 > 0 && mtype2 == 0 && brchtype1 == Branch::_BASALDEN)
    //if (mtype1 > 0 && mtype2 == 0)
  {
    rval = true;
    key = tkey1;
    //std::cout << brchtype1+1 << brchtype2+1<< " ";
    assert(mtype1 == 2 or mtype1 == 3);
  }
  if (mtype1 == 0 && mtype2 > 0 && brchtype2 == Branch::_BASALDEN)
    //if (mtype1 == 0 && mtype2 > 0)
  {
    rval = true;
    key = tkey2;
    assert(mtype2 == 2 or mtype2 == 3);
    //std::cout << brchtype2+1  << brchtype1+1<< " ";
  }

  return rval;

}
#ifdef SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM
bool Touch::hasSpineNeck(key_size_t& key, Params& params)
{
  //TUAN TODO POTENTIAL BUG
  //right now, for single neuron scenario, 
  // if one-side MTYPE > 0 and the other-side MTYPE=0
  // indicate a spine-denshaft touch
  //     
  bool rval=false;

  key_size_t tkey1 = getKey1();
  //NOTE: uf0 = MTYPE
  unsigned int mtype1 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey1);
  unsigned int brchtype1 = _segmentDescriptor.getBranchType(tkey1);
  key_size_t tkey2 = getKey2();
  unsigned int mtype2 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey2);
  unsigned int brchtype2 = _segmentDescriptor.getBranchType(tkey2);

  if (params.isGivenKeySpineNeck(tkey1))
    //if (mtype1 > 0 && mtype2 == 0 && brchtype1 == Branch::_BASALDEN)
    //if (mtype1 > 0 && mtype2 == 0)
  {
    rval = true;
    key = tkey1;
    //std::cout << brchtype1+1 << brchtype2+1<< " ";
  }
  if (params.isGivenKeySpineNeck(tkey2))
    //if (mtype1 == 0 && mtype2 > 0 && brchtype2 == Branch::_BASALDEN)
    //if (mtype1 == 0 && mtype2 > 0)
  {
    rval = true;
    key = tkey2;
    //std::cout << brchtype2+1  << brchtype1+1<< " ";
  }

  return rval;

}

bool Touch::isSpineNeck_n_DenShaft(key_size_t& key, Params& params)
{
  //TUAN TODO POTENTIAL BUG
  //right now, for single neuron scenario, 
  // if one-side MTYPE > 0 and the other-side MTYPE=0
  // indicate a spine-denshaft touch
  //     
  bool rval=false;

  key_size_t tkey1 = getKey1();
  //NOTE: uf0 = MTYPE
  unsigned int mtype1 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey1);
  unsigned int brchtype1 = _segmentDescriptor.getBranchType(tkey1);
  key_size_t tkey2 = getKey2();
  unsigned int mtype2 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey2);
  unsigned int brchtype2 = _segmentDescriptor.getBranchType(tkey2);

  if (params.isGivenKeySpineNeck(tkey1) && 
      (brchtype1 == Branch::_BASALDEN || 
       brchtype1 == Branch::_APICALDEN || 
       brchtype1 == Branch::_TUFTEDDEN  
      )
     )
  {
    rval = true;
    key = tkey1;
    //std::cout << brchtype1+1 << brchtype2+1<< " ";
  }
  if (params.isGivenKeySpineNeck(tkey2) && 
      (brchtype1 == Branch::_BASALDEN || 
       brchtype1 == Branch::_APICALDEN || 
       brchtype1 == Branch::_TUFTEDDEN  
      )
     )
  {
    rval = true;
    key = tkey2;
    //std::cout << brchtype2+1  << brchtype1+1<< " ";
  }

  return rval;

}
#endif


//GOAL: 
// A touch has 2 capsule
//    check if one of the two capsules is the spinehead
//    and return the key of that capsule
// TODO: change the way we determine in the coming future
//       where the spine head+neck is not passed in via tissue file
//       but automatically generated
bool Touch::hasSpineHead(key_size_t& key)//obsolete
{
	//right now, for single neuron scenario, 
	// a spine-bouton touch
	// if 
	//   1. one-side MTYPE > 0 and the other-side MTYPE>0
	//   2. the neuron index difference == 1
	//     
	bool rval=false;

	key_size_t tkey1 = getKey1();
	//NOTE: uf0 = MTYPE
	unsigned int mtype1 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey1);
	key_size_t tkey2 = getKey2();
	unsigned int mtype2 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey2);
	unsigned int brtype1 = _segmentDescriptor.getValue(SegmentDescriptor::branchType, tkey1);
	unsigned int brtype2 = _segmentDescriptor.getValue(SegmentDescriptor::branchType, tkey2);
	unsigned int neuronIdx1 = _segmentDescriptor.getValue(SegmentDescriptor::neuronIndex, tkey1);
	unsigned int neuronIdx2 = _segmentDescriptor.getValue(SegmentDescriptor::neuronIndex, tkey2);
	bool isChemSynapsePair = (brtype1 == Branch::_AXON and brtype2 == Branch::_SOMA) ||
		(brtype2 == Branch::_AXON and brtype1 == Branch::_SOMA) ;
	if (mtype1 > 0 && mtype2 > 0 and std::abs((int)neuronIdx1-(int)neuronIdx2) ==1 and
	  isChemSynapsePair)
	{
		rval = true;
		if (brtype1 == Branch::_AXON)
			key =  tkey2;
		else
			key = tkey1;
	}
	return rval;
}

#ifdef SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM
bool Touch::hasSpineHead(key_size_t& key, Params& params) 
{
	//right now, for single neuron scenario, 
	// a spine-bouton touch
	// if 
	//   1. one-side MTYPE > 0 and the other-side MTYPE>0
	//   2. the neuron index difference == 1
	//     
	bool rval=false;

	key_size_t tkey1 = getKey1();
	//NOTE: uf0 = MTYPE
	unsigned int mtype1 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey1);
	key_size_t tkey2 = getKey2();
	unsigned int mtype2 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey2);
	unsigned int brtype1 = _segmentDescriptor.getValue(SegmentDescriptor::branchType, tkey1);
	unsigned int brtype2 = _segmentDescriptor.getValue(SegmentDescriptor::branchType, tkey2);
	unsigned int neuronIdx1 = _segmentDescriptor.getValue(SegmentDescriptor::neuronIndex, tkey1);
	unsigned int neuronIdx2 = _segmentDescriptor.getValue(SegmentDescriptor::neuronIndex, tkey2);
	bool isChemSynapsePair = (brtype1 == Branch::_AXON and brtype2 == Branch::_SOMA) ||
		(brtype2 == Branch::_AXON and brtype1 == Branch::_SOMA) ;
  if (params.isGivenKeySpineHead(tkey1))
	//if (mtype1 > 0 && mtype2 > 0 and std::abs((int)neuronIdx1-(int)neuronIdx2) ==1 and
	//  isChemSynapsePair)
	{
		rval = true;
    key = tkey1;
	}
  if (params.isGivenKeySpineHead(tkey2))
  {
		rval = true;
    key = tkey2;
  }
	return rval;
}
#endif

//GOAL: 
// A touch has 2 capsule
//     check if it is a spineless touch that can form a chemical synapse
// RETURN:
//  true =  if it is a spineless touch 
//  axon_key = the key of the capsule that serves as the axonic-side
// TUAN TODO: change the way we determine in the coming future
//       where the spine head+neck is not passed in via tissue file
//       but automatically generated
bool Touch::isSpineless(key_size_t& axon_key)
{
	bool rval=false;

	key_size_t tkey1 = getKey1();
	//NOTE: uf0 = MTYPE
	unsigned int mtype1 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey1);
	key_size_t tkey2 = getKey2();
	unsigned int mtype2 = _segmentDescriptor.getValue(SegmentDescriptor::uf0, tkey2);
	unsigned int brtype1 = _segmentDescriptor.getValue(SegmentDescriptor::branchType, tkey1);
	unsigned int brtype2 = _segmentDescriptor.getValue(SegmentDescriptor::branchType, tkey2);
	unsigned int neuronIdx1 = _segmentDescriptor.getValue(SegmentDescriptor::neuronIndex, tkey1);
	unsigned int neuronIdx2 = _segmentDescriptor.getValue(SegmentDescriptor::neuronIndex, tkey2);
	bool isSpinelessChemSynapsePair = (brtype1 == Branch::_AXON and (brtype2 == Branch::_BASALDEN ||
				brtype2 == Branch::_APICALDEN || brtype2 == Branch::_TUFTEDDEN)
			) ||
		(brtype2 == Branch::_AXON and 
		 (brtype1 == Branch::_BASALDEN ||
				brtype1 == Branch::_APICALDEN || brtype1 == Branch::_TUFTEDDEN)
		 ) ;
	if (isSpinelessChemSynapsePair)
//mtype1 > 0 && mtype2 > 0 and std::abs((int)neuronIdx1-(int)neuronIdx2) ==1 and
	{
		rval = true;
		if (brtype1 == Branch::_AXON)
			axon_key =  tkey1;
		else
			axon_key = tkey2;
	}
	return rval;
}

Touch::~Touch() {}

Touch::compare::compare(int c) : _case(c) {}

bool Touch::compare::operator()(const Touch& t0, const Touch& t1) {
  bool rval = false;

  key_size_t key0, key1, key2, key3;
  if (_case == 0) {
    key0 = t0._touchData[0];
    key1 = t1._touchData[0];
    key2 = t0._touchData[1];
    key3 = t1._touchData[1];
  } else if (_case == 1) {
    key0 = t0._touchData[1];
    key1 = t1._touchData[1];
    key2 = t0._touchData[0];
    key3 = t1._touchData[0];
  } else
    assert(0);

  unsigned int n0 = _segmentDescriptor.getNeuronIndex(key0);
  unsigned int n1 = _segmentDescriptor.getNeuronIndex(key1);

  if (n0 == n1) {
    unsigned int b0 = _segmentDescriptor.getBranchIndex(key0);
    unsigned int b1 = _segmentDescriptor.getBranchIndex(key1);

    if (b0 == b1) {
      unsigned int s0 = _segmentDescriptor.getSegmentIndex(key0);
      unsigned int s1 = _segmentDescriptor.getSegmentIndex(key1);

      if (s0 == s1) {
        unsigned int n2 = _segmentDescriptor.getNeuronIndex(key2);
        unsigned int n3 = _segmentDescriptor.getNeuronIndex(key3);

        if (n2 == n3) {
          unsigned int b2 = _segmentDescriptor.getBranchIndex(key2);
          unsigned int b3 = _segmentDescriptor.getBranchIndex(key3);

          if (b2 == b3) {
            unsigned int s2 = _segmentDescriptor.getSegmentIndex(key2);
            unsigned int s3 = _segmentDescriptor.getSegmentIndex(key3);

            rval = (s2 < s3);

          } else
            rval = (b2 < b3);

        } else
          rval = (n2 < n3);

      } else
        rval = (s0 < s1);

    } else
      rval = (b0 < b1);

  } else
    rval = (n0 < n1);

  return rval;
}

Touch& Touch::operator=(const Touch& t) {
  if (this == &t) return *this;
  std::copy(t._touchData, t._touchData + N_TOUCH_DATA, _touchData);
// memcpy(_touchData, t._touchData, N_TOUCH_DATA*sizeof(double));
#ifndef LTWT_TOUCH
  std::copy(t._endTouch, t._endTouch + 4, _endTouch);
  // memcpy(_endTouch, t._endTouch, 4*sizeof(short));
  _remains = t._remains;
#endif
  return *this;
}

bool Touch::operator==(const Touch& t) {
  if (this == &t) return true;
  key_size_t key0 = _touchData[0];
  key_size_t key1 = t._touchData[0];
  key_size_t key2 = _touchData[1];
  key_size_t key3 = t._touchData[1];
  return ((key0 == key1 && key2 == key3));  //|| (key0==key3 && key1==key2) );
}
