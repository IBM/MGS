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

#ifndef SEGMENTDESCRIPTOR_H
#define SEGMENTDESCRIPTOR_H

#include "../../nti/include/MaxComputeOrder.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include <string>
#include <vector>
#include <cassert>
#include <map>

#define _STRG(x) #x
#define STRG(x) _STRG(x)

#define N_FIELDS 9

#define FIELD_0 MTYPE
#define FIELD_1 ETYPE
#define FIELD_2 BRANCHTYPE
#define FIELD_3 BRANCHORDER
#define FIELD_4 COMPUTEORDER
#define FIELD_5 SEGMENT_INDEX
#define FIELD_6 NEURON_INDEX
#define FIELD_7 BRANCH_INDEX
#define FIELD_8 FLAG

#define USER_FIELD_0_BITS 4    // < 8 morphological types
//#define USER_FIELD_1_BITS 4    // < 16 electrophysiological types
//#define BRANCH_TYPE_BITS 2     // 4 types (soma, axon, dend, apic)
#define USER_FIELD_1_BITS 3    // < 8 electrophysiological types
#define BRANCH_TYPE_BITS 3     // 8 types (soma, axon, (basal)dend, apic, AIS, tufts, axonhillock, bouton)
#define BRANCH_ORDER_BITS 7    // < 128 branch order
#define COMPUTE_ORDER_BITS 3   // 0-7 compute order
#define SEGMENT_INDEX_BITS 12  // < 4,096 segments/branch
#define NEURON_INDEX_BITS 19   // < 524,288 neurons
#define BRANCH_INDEX_BITS 12   // < 4,096 branches/neuron
#define FLAG_BITS 1            // transient volatile boolean (true if the 
                     //associated compartment is a junction)

class Segment;
//TUAN: need to update all possible type to key_size_t
//      and use an equivalent name for 'unsigned long long'

/**
 * assign a unique id for a given segment object, or rather return a
 * 'second-order' segment id for a given id and a certain criterion,
 * meanwhile able to retrieve the respective information stored in the id
 * (e.g. eType, mType, etc.)
 */
class SegmentDescriptor {
  public:
  enum SegmentKeyData {
    uf0 = 0,
    uf1,
    branchType,
    branchOrder,
    computeOrder,
    segmentIndex,
    neuronIndex,
    branchIndex,
    flag
  };

  SegmentDescriptor();
  SegmentDescriptor(SegmentDescriptor const&);
  SegmentDescriptor(SegmentDescriptor&);
  ~SegmentDescriptor();
  key_size_t flipFlag(key_size_t key);
  /* give the abstract key based on the vector of chosen key-fields */
  unsigned long long getMask(std::vector<SegmentKeyData> const& maskVector);
  /* give the true key for a segment */
  key_size_t getSegmentKey(Segment* segment);
  /* give the abstract key given the values for each given key-field */
  key_size_t getSegmentKey(std::vector<SegmentKeyData> const& maskVector,
                           unsigned int* ids);
  /* give the abstract key from the real key by 
   * picking out the values present in the 'mask' */
  key_size_t getSegmentKey(key_size_t segmentKey, unsigned long long mask);
  key_size_t modifySegmentKey(SegmentKeyData uf, unsigned int id,
                              key_size_t key);
  void printKey(key_size_t segmentKey, unsigned long long mask);
  void printMask(unsigned long long mask);
  unsigned long long getLongKey(key_size_t key);

  unsigned int getValue(SegmentKeyData skd, key_size_t segmentKey);
  SegmentKeyData getSegmentKeyData(std::string fieldName);
  unsigned int getBranchType(key_size_t segmentKey);
  unsigned int getBranchOrder(key_size_t segmentKey);
  unsigned int getComputeOrder(key_size_t segmentKey);
  unsigned int getNeuronIndex(key_size_t segmentKey);
  unsigned int getBranchIndex(key_size_t segmentKey);
  unsigned int getSegmentIndex(key_size_t segmentKey);
  bool getFlag(key_size_t segmentKey);
  std::string getFieldName(const int& fieldValue);

  private:
  static unsigned int pow2(int n);

  /**
   * 64bit unsigned long long field for segment id, MUST be divided into 2 32bit
   * parts,
   * in order for the bitmasking to function properly, so we still need to
   * agree on one best bit allocation pattern
   */
  struct segmentKeyBitPattern {
    unsigned int userField0 : USER_FIELD_0_BITS;
    unsigned int userField1 : USER_FIELD_1_BITS;
    unsigned int branchType : BRANCH_TYPE_BITS;
    unsigned int branchOrder : BRANCH_ORDER_BITS;
    unsigned int computeOrder : COMPUTE_ORDER_BITS;
    unsigned int segmentIndex : SEGMENT_INDEX_BITS;
    unsigned int neuronIndex : NEURON_INDEX_BITS;
    unsigned int branchIndex : BRANCH_INDEX_BITS;
    unsigned int flag : FLAG_BITS;
  };

  /**
   * union for segment id, simultaneously holds an unsigned long long, a double
   * or a struct
   */
  union SegmentID {
    unsigned long long idLong;
    segmentKeyBitPattern key;
    double collapsed;
  };

  unsigned long long _mask[N_FIELDS];
  SegmentID sid;
  int _maxField[N_FIELDS];

  std::vector<std::string> _fields;
  std::map<std::string, SegmentKeyData> _fieldMap;
};

#endif
