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

#ifndef SEGMENTDESCRIPTOR_H
#define SEGMENTDESCRIPTOR_H

#include <mpi.h>
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

#define USER_FIELD_0_BITS 4   // < 8 morphological types
#define USER_FIELD_1_BITS 4   // < 8 electrophysiological types
#define BRANCH_TYPE_BITS 2    // 4 types (soma, axon, dend, apic)
#define BRANCH_ORDER_BITS 7   // < 128 branch order
#define COMPUTE_ORDER_BITS 3  // 0-7 compute order
#define SEGMENT_INDEX_BITS 12 // < 4,096 segments/branch
#define NEURON_INDEX_BITS 19  // < 524,288 neurons
#define BRANCH_INDEX_BITS 12   // < 4,096 branches/neuron
#define FLAG_BITS 1           // transient volatile boolean

class Segment;

/**
 * assign a unique id for a given segment object, or rather return a 
 * 'second-order' segment id for a given id and a certain criterion,
 * meanwhile able to retrieve the respective information stored in the id
 * (e.g. eType, mType, etc.)
 */
class SegmentDescriptor
{
   public:
           
      enum SegmentKeyData {
	uf0=0,
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
      SegmentDescriptor(SegmentDescriptor const &);
      SegmentDescriptor(SegmentDescriptor&);
      ~SegmentDescriptor();
      double getSegmentKey(Segment* segment);
      double flipFlag(double key);
      double getSegmentKey(std::vector<SegmentKeyData> const & maskVector, unsigned int* ids);
      unsigned long long getMask(std::vector<SegmentKeyData> const & maskVector);
      double getSegmentKey(double segmentKey, unsigned long long mask);
      double modifySegmentKey(SegmentKeyData uf, unsigned int id, double key);
      void printKey(double segmentKey, unsigned long long mask);
      void printMask(unsigned long long mask);
      unsigned long long getLongKey(double key);

      unsigned int getValue(SegmentKeyData skd, double segmentKey);
      SegmentKeyData getSegmentKeyData(std::string fieldName);
      unsigned int getBranchType(double segmentKey);
      unsigned int getBranchOrder(double segmentKey);
      unsigned int getComputeOrder(double segmentKey);      
      unsigned int getNeuronIndex (double segmentKey);
      unsigned int getBranchIndex(double segmentKey);
      unsigned int getSegmentIndex(double segmentKey);
      bool getFlag(double segmentKey);

   private:
      static unsigned int pow2(int n);

     /**
      * 64bit unsigned long long field for segment id, MUST be divided into 2 32bit parts,
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
