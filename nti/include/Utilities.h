// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
//     Datatype datatype(2, &className, MPI_Aint lowerBound, MPI_Aint extent);
//     datatype.set(0, MPI_DOUBLE, 1, &className.placeHolderA);
//     datatype.set(1, MPI_DOUBLE, 1, &className.placeHolderB);
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
		MPI_Aint lowerBound;
		MPI_Aint extent;
		// Private and no definition...
		Datatype(const Datatype &);
		Datatype &operator=(const Datatype &);
		static MPI_Aint getAddress(void *pointer);
	public:
		Datatype(int n, void *start, MPI_Aint lowerBound, MPI_Aint extent);
		virtual ~Datatype();
		//
		void set(int i, MPI_Datatype datatype, int length, void *address);
		MPI_Datatype create();
		MPI_Datatype commit();
		operator MPI_Datatype();
};

inline MPI_Aint Datatype::getAddress(void *pointer) {
	MPI_Aint address;
	MPI_Get_address(pointer, &address);
	return(address);
}
inline Datatype::Datatype(int n, void *start, MPI_Aint lb, MPI_Aint ext) {
	fieldN = n;
	fieldStart = start;
	fieldLengths = new int[n];
	fieldDisplacements = new MPI_Aint[n];
	fieldDatatypes = new MPI_Datatype[n];
	fieldDatatype = MPI_DATATYPE_NULL;
	fieldCommitted = false;
	lowerBound = lb;
	extent = ext;
}
inline Datatype::~Datatype() {
	delete[] (fieldLengths);
	delete[] (fieldDisplacements);
	delete[] (fieldDatatypes);
}
inline void Datatype::set(int i, MPI_Datatype datatype, int length, void *address) {
	fieldDatatypes[i] = datatype;
	fieldLengths[i] = length;
	fieldDisplacements[i] = getAddress(address) - getAddress(fieldStart);
	//std::cout << "Displacement[" << i << "] = " << fieldDisplacements[i] << "\n";
}
inline MPI_Datatype Datatype::create() {
	if (fieldDatatype == MPI_DATATYPE_NULL) {
	  MPI_Datatype tmpType;
	  MPI_Type_create_struct(fieldN, fieldLengths, fieldDisplacements, fieldDatatypes, &tmpType);
	  MPI_Type_create_resized(tmpType, lowerBound, extent, &fieldDatatype);
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
