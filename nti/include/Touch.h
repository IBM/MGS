// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef TOUCH_H
#define TOUCH_H

#include <mpi.h>
#include <vector>
#include <string.h>

#include "SegmentDescriptor.h"
#include "NTSMacros.h"
#include "MaxComputeOrder.h"

//#define LTWT_TOUCH (moved to NTSMacro.h)

#ifndef LTWT_TOUCH
#define N_TOUCH_DATA 5
#else
#define N_TOUCH_DATA 4
#endif

class Touch
{
  friend class TissueContext;

 public:
  Touch();
  Touch(Touch const &);
  Touch& operator=(const Touch& touch);
  bool operator==(const Touch& touch);
  static MPI_Datatype* getTypeTouch();
  ~Touch();
		
  //TUAN: NOTE - potential bug here if we change key's size
  //key_size_t 
  //TODO- modify to move the two key parts to the end
  //and use offset = sizeof(key_size_t)/sizeof(double)
  //e.g. _touchData[FIRST_KEY_INDEX], _touchData[FIRST_KEY_INDEX+offset]
  double* getTouchData() {return _touchData;}
  double* getKeys() {return _touchData;}
  double* getProps() {return &_touchData[2];}

  key_size_t getKey1() {return _touchData[0];}
  key_size_t getKey2() {return _touchData[1];}
  void setKey1(key_size_t key1) {_touchData[0]=key1;}
  void setKey2(key_size_t key2) {_touchData[1]=key2;}
  void setProp1(double prop1) {_touchData[2]=prop1;}
  void setProp2(double prop2) {_touchData[3]=prop2;}

  key_size_t getPartner(key_size_t key);

  void readFromFile(FILE* dataFile);
  void writeToFile(FILE* dataFile);
  void printTouch();

#ifndef LTWT_TOUCH
  double getDistance() const {return _touchData[4];}
  void setDistance(double distance) {_touchData[4]=distance;}

  short* getEndTouches() {return _endTouch;}
  short getEndTouch1() {return _endTouch[0];}
  short getEndTouch2() {return _endTouch[1];}
  short getEndTouch3() {return _endTouch[2];}
  short getEndTouch4() {return _endTouch[3];}

  bool remains() {return _remains;}
  void eliminate() {_remains=false;}
  void reinstate() {_remains=true;}
#endif

  class compare
    {
    public:
      compare(int c);
      bool operator()(const Touch& t0, const Touch& t1);
    private:
      int _case;
    };


	//return: sc (the scaling term in [0,1] form the starting coord on the first capsule 
	//           at that the point with minimum distance locates)
  double getProp1() {return _touchData[2];}
	//return: sc (the scaling term in [0,1] form the starting coord on the second capsule
	//           at that the point with minimum distance locates)
  double getProp2() {return _touchData[3];}
  double getProp(double key);

	//if the touch has one side is spine-neck then return 'true'
	//  and pass the key of the capsule as spine-neck out via argument
	bool hasSpineNeck(key_size_t& key);
	//if the touch has one side is spine-head then return 'true'
	//  and pass the key of the capsule as spine-head out via argument
	bool hasSpineHead(key_size_t& key);
 private:
  // [0] = key1
	// [1] = key2
	// [2] = prop1
	// [3] = prop2
	// [4] = distance of 2 capsules forming the touch
  double _touchData[N_TOUCH_DATA];

#ifndef LTWT_TOUCH
  short _endTouch[4]; // end1A, end1B, end2A, end2B
  bool _remains;
#endif		

  // this is used to hold data for a Touch, but organized in such a way that can be used
  // for data exchange between MPI processes 
  static MPI_Datatype* _typeTouch;
  static SegmentDescriptor _segmentDescriptor;
};

#endif
