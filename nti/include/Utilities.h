// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

// An example of the use of this class...
// MPI_Datatype ClassName::fieldDatatype = MPI_DATATYPE_NULL;
// MPI_Datatype ClassName::getDatatype() {
//   if (fieldDatatype == MPI_DATATYPE_NULL) {
//     ClassName className;
//     Datatype datatype(4, &className);
//     datatype.set(0, MPI_LB, 0);
//     datatype.set(1, MPI_DOUBLE, 1, &className.placeHolderA);
//     datatype.set(2, MPI_DOUBLE, 1, &className.placeHolderB);
//     datatype.set(3, MPI_UB, sizeof(ClassName));
//     fieldDatatype = datatype.commit();
//   }
//   return(fieldDatatype);
// }

class Datatype {
	private:
		int fieldN;
		void *fieldStart;
		int *fieldLengths;
		MPI_Aint *fieldDisplacements;
		MPI_Datatype *fieldDatatypes;
		MPI_Datatype fieldDatatype;
		bool fieldCommitted;
		// Private and no definition...
		Datatype(const Datatype &);
		Datatype &operator=(const Datatype &);
		static MPI_Aint getAddress(void *pointer);
	public:
		Datatype(int n, void *start);
		virtual ~Datatype();
		//
		void set(int i, MPI_Datatype datatype, MPI_Aint displacement);
		void set(int i, MPI_Datatype datatype, int length, void *address);
		MPI_Datatype create();
		MPI_Datatype commit();
		operator MPI_Datatype();
};

inline MPI_Aint Datatype::getAddress(void *pointer) {
	MPI_Aint address;
	MPI_Address(pointer, &address);
	return(address);
}
inline Datatype::Datatype(int n, void *start) {
	fieldN = n;
	fieldStart = start;
	fieldLengths = new int[n];
	fieldDisplacements = new MPI_Aint[n];
	fieldDatatypes = new MPI_Datatype[n];
	fieldDatatype = MPI_DATATYPE_NULL;
	fieldCommitted = false;
}
inline Datatype::~Datatype() {
	delete[] (fieldLengths);
	delete[] (fieldDisplacements);
	delete[] (fieldDatatypes);
}
inline void Datatype::set(int i, MPI_Datatype datatype, MPI_Aint displacement) {
	fieldDatatypes[i] = datatype;
	fieldLengths[i] = 1;
	fieldDisplacements[i] = displacement;
	//std::cout << "Displacement[" << i << "] = " << fieldDisplacements[i] << "\n";
}
inline void Datatype::set(int i, MPI_Datatype datatype, int length, void *address) {
	fieldDatatypes[i] = datatype;
	fieldLengths[i] = length;
	fieldDisplacements[i] = getAddress(address) - getAddress(fieldStart);
	//std::cout << "Displacement[" << i << "] = " << fieldDisplacements[i] << "\n";
}
inline MPI_Datatype Datatype::create() {
	if (fieldDatatype == MPI_DATATYPE_NULL) {
		MPI_Type_struct(fieldN, fieldLengths, fieldDisplacements, fieldDatatypes, &fieldDatatype);
	}
	return(fieldDatatype);
}
inline MPI_Datatype Datatype::commit() {
	create();
	if (fieldCommitted == false) {
		MPI_Type_commit(&fieldDatatype);
		fieldCommitted = true;
	}
	return(fieldDatatype);
}
inline Datatype::operator MPI_Datatype() {
	return(fieldDatatype);
}
#endif /*UTILITIES_H_*/
