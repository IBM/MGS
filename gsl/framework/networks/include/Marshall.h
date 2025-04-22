// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MARSHALL_H
#define MARSHALL_H
#include "Copyright.h"
#ifdef HAVE_MPI
#include <mpi.h>
#include "OutputStream.h"
#include "Array.h"
#include "ShallowArray.h"
#include "DeepPointerArray.h"
#include "NDPairList.h"
#include <string>
#include <iostream>
#include <vector>

template <class T>
class MarshallerInstance 
{
public:
   void marshall(OutputStream* stream, T const& data) {
      *stream << data;
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, T const& data) {
     blengths.push_back(sizeof(T));
     MPI_Aint blockAddress;
     MPI_Get_address(const_cast<T*>(&data), &blockAddress);
     blocs.push_back(blockAddress);
   }
};

template <>
class MarshallerInstance<std::string>
{
public:
   void marshall(OutputStream* stream, std::string const& data) {
     assert(0); // Need to create our own string type to communicate size like Arrays: JK
     int s = data.size();
     *stream << s;
     *stream << data;
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, std::string const& data) {
     assert(0); // Need to create our own string type to communicate size like Arrays: JK
     blengths.push_back(data.size());
     MPI_Aint blockAddress;
     MPI_Get_address(const_cast<char*>(data.c_str()), &blockAddress);
     blocs.push_back(blockAddress);
   }
};

template <>
class MarshallerInstance<NDPairList*>
{
public:
   void marshall(OutputStream* stream, NDPairList* data) {
     assert(0);  // Need to create our own string type to communicate size like Arrays: JK
     *stream << *data;
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, NDPairList const * data) {
     assert(0);  // Need to create our own string type to communicate size like Arrays: JK
   }
};

template <class T>
class MarshallerInstance<Array<T> >
{
public:
  /* TODO TUAN check if we need to modify to work with flat array for GPU here */
   void marshall(OutputStream* stream, Array<T> const & data) {
     MarshallerInstance<T> mi1;
     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
     for (unsigned i=0; i < sz; ++i)
       mi1.marshall(stream, data[i]);
     if (sz<data.getCommunicatedSize()) {
       T padding{};
       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
	      mi1.marshall(stream, padding);
     }
     MarshallerInstance<unsigned> mi2;
     mi2.marshall(stream, data.size());
     if (data.size() != data.getCommunicatedSize()) {
       const_cast<Array<T>*>(&data)->setSizeToCommunicate(data.size());
       stream->requestRebuild(true);
     }
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, Array<T> const & data) {
     MarshallerInstance<T> mi1;
     for (unsigned i=0; i < data.getSizeToCommunicate(); ++i)
       mi1.getBlocks(blengths, blocs, data[i]);
     MarshallerInstance<unsigned> mi2;
     mi2.getBlocks(blengths, blocs, data.getSizeToCommunicate());
     const_cast<Array<T>*>(&data)->setCommunicatedSize(data.getSizeToCommunicate());
   }
};

template <class T>
class MarshallerInstance<ShallowArray<T> >
{
public:
   void marshall(OutputStream* stream, ShallowArray<T> const & data) {
     MarshallerInstance<T> mi1;
     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
     for (unsigned i=0; i < sz; ++i)
       mi1.marshall(stream, data[i]);
     if (sz<data.getCommunicatedSize()) {
       T padding{};
       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
	      mi1.marshall(stream, padding);
     }
     MarshallerInstance<unsigned> mi2;
     sz = data.size();
     mi2.marshall(stream, sz);
     if (sz != data.getCommunicatedSize()) {
       const_cast<ShallowArray<T>*>(&data)->setSizeToCommunicate(sz);
       stream->requestRebuild(true);
     }
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, ShallowArray<T> const & data) {
     MarshallerInstance<T> mi1;
     for (unsigned i=0; i < data.getSizeToCommunicate(); ++i)
       mi1.getBlocks(blengths, blocs, data[i]);
     MarshallerInstance<unsigned> mi2;
     mi2.getBlocks(blengths, blocs, data.getSizeToCommunicate());
     const_cast<ShallowArray<T>*>(&data)->setCommunicatedSize(data.getSizeToCommunicate());
   }
};

//TUAN TODO: check if we need to make a copy for this with Array::UNIFIED_MEM
template <class T>
class MarshallerInstance<ShallowArray<T,3,2> >
{
public:
  void marshall(OutputStream* stream, ShallowArray<T,3,2> const & data) {
     MarshallerInstance<T> mi1;
     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
     for (unsigned i=0; i < sz; ++i)
       mi1.marshall(stream, data[i]);
     if (sz<data.getCommunicatedSize()) {
       T padding{};
       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
	      mi1.marshall(stream, padding);
     }
     MarshallerInstance<unsigned> mi2;
     mi2.marshall(stream, data.size());
     if (data.size() != data.getCommunicatedSize()) {
       const_cast<ShallowArray<T,3,2>*>(&data)->setSizeToCommunicate(data.size());
       stream->requestRebuild(true);
     }
   }
  void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, ShallowArray<T,3,2> const & data) {
     MarshallerInstance<T> mi1;
     for (unsigned i=0; i < data.getSizeToCommunicate(); ++i)
       mi1.getBlocks(blengths, blocs, data[i]);
     MarshallerInstance<unsigned> mi2;
     mi2.getBlocks(blengths, blocs, data.getSizeToCommunicate());
     const_cast<ShallowArray<T,3,2>*>(&data)->setCommunicatedSize(data.getSizeToCommunicate());
   }
};

template <class T>
class MarshallerInstance<DeepPointerArray<T> >
{
public:
   void marshall(OutputStream* stream, DeepPointerArray<T> const & data) {
     MarshallerInstance<T> mi1;
     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
     for (unsigned i=0; i < sz; ++i)
       mi1.marshall(stream, data[i]);
     if (sz<data.getCommunicatedSize()) {
       T padding{};
       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
	      mi1.marshall(stream, padding);
     }
     MarshallerInstance<unsigned> mi2;
     mi2.marshall(stream, data.size());
     if (data.size() != data.getCommunicatedSize()) {
       const_cast<DeepPointerArray<T>*>(&data)->setSizeToCommunicate(data.size());
       stream->requestRebuild(true);
     }
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, DeepPointerArray<T> const & data) {
     MarshallerInstance<T> mi1;
     for (unsigned i=0; i < data.getSizeToCommunicate(); ++i)
       mi1.getBlocks(blengths, blocs, data[i]);
     MarshallerInstance<unsigned> mi2;
     mi2.getBlocks(blengths, blocs, data.getSizeToCommunicate());
     const_cast<DeepPointerArray<T>*>(&data)->setCommunicatedSize(data.getSizeToCommunicate());
   }
};

//#if defined(HAVE_GPU) && defined(__NVCC__)
//template <class T>
//class MarshallerInstance<Array<T, Array::UNIFIED_MEM> >
//{
//public:
//  /* TODO TUAN check if we need to modify to work with flat array for GPU here */
//   void marshall(OutputStream* stream, Array<T, Array::UNIFIED_MEM> const & data) {
//     MarshallerInstance<T> mi1;
//     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
//     for (unsigned i=0; i < sz; ++i)
//       mi1.marshall(stream, data[i]);
//     if (sz<data.getCommunicatedSize()) {
//       T padding;
//       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
//	 mi1.marshall(stream, padding);
//     }
//     MarshallerInstance<unsigned> mi2;
//     mi2.marshall(stream, data.size());
//     if (data.size() != data.getCommunicatedSize()) {
//       const_cast<Array<T, Array::UNIFIED_MEM>*>(&data)->setSizeToCommunicate(data.size());
//       stream->requestRebuild(true);
//     }
//   }
//   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, Array<T, Array::UNIFIED_MEM> const & data) {
//     MarshallerInstance<T> mi1;
//     for (unsigned i=0; i < data.getSizeToCommunicate(); ++i)
//       mi1.getBlocks(blengths, blocs, data[i]);
//     MarshallerInstance<unsigned> mi2;
//     mi2.getBlocks(blengths, blocs, data.getSizeToCommunicate());
//     const_cast<Array<T, Array::UNIFIED_MEM>*>(&data)->setCommunicatedSize(data.getSizeToCommunicate());
//   }
//};
//
//template <class T>
//class MarshallerInstance<ShallowArray<T, Array::UNIFIED_MEM> >
//{
//public:
//   void marshall(OutputStream* stream, ShallowArray<T, Array::UNIFIED_MEM> const & data) {
//     MarshallerInstance<T> mi1;
//     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
//     for (unsigned i=0; i < sz; ++i)
//       mi1.marshall(stream, data[i]);
//     if (sz<data.getCommunicatedSize()) {
//       T padding;
//       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
//	 mi1.marshall(stream, padding);
//     }
//     MarshallerInstance<unsigned> mi2;
//     sz = data.size();
//     mi2.marshall(stream, sz);
//     if (sz != data.getCommunicatedSize()) {
//       const_cast<ShallowArray<T, Array::UNIFIED_MEM>*>(&data)->setSizeToCommunicate(sz);
//       stream->requestRebuild(true);
//     }
//   }
//   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, ShallowArray<T, Array::UNIFIED_MEM> const & data) {
//     MarshallerInstance<T> mi1;
//     for (unsigned i=0; i < data.getSizeToCommunicate(); ++i)
//       mi1.getBlocks(blengths, blocs, data[i]);
//     MarshallerInstance<unsigned> mi2;
//     mi2.getBlocks(blengths, blocs, data.getSizeToCommunicate());
//     const_cast<ShallowArray<T, Array::UNIFIED_MEM>*>(&data)->setCommunicatedSize(data.getSizeToCommunicate());
//   }
//};
//
//
//template <class T>
//class MarshallerInstance<DeepPointerArray<T, Array::UNIFIED_MEM> >
//{
//public:
//   void marshall(OutputStream* stream, DeepPointerArray<T, Array::UNIFIED_MEM> const & data) {
//     MarshallerInstance<T> mi1;
//     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
//     for (unsigned i=0; i < sz; ++i)
//       mi1.marshall(stream, data[i]);
//     if (sz<data.getCommunicatedSize()) {
//       T padding;
//       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
//	 mi1.marshall(stream, padding);
//     }
//     MarshallerInstance<unsigned> mi2;
//     mi2.marshall(stream, data.size());
//     if (data.size() != data.getCommunicatedSize()) {
//       const_cast<DeepPointerArray<T, Array::UNIFIED_MEM>*>(&data)->setSizeToCommunicate(data.size());
//       stream->requestRebuild(true);
//     }
//   }
//   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, DeepPointerArray<T, Array:UNIFIED_MEM> const & data) {
//     MarshallerInstance<T> mi1;
//     for (unsigned i=0; i < data.getSizeToCommunicate(); ++i)
//       mi1.getBlocks(blengths, blocs, data[i]);
//     MarshallerInstance<unsigned> mi2;
//     mi2.getBlocks(blengths, blocs, data.getSizeToCommunicate());
//     const_cast<DeepPointerArray<T, Array::UNIFIED_MEM>*>(&data)->setCommunicatedSize(data.getSizeToCommunicate());
//   }
//};
//#endif
/*
template <class T>
class MarshallerInstance<ShallowArray<T> >
{
public:
   void marshall(OutputStream* stream, ShallowArray<T> const & data) {
     MarshallerInstance<T> mi1;
     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
     for (unsigned i=0; i < sz; ++i)
       mi1.marshall(stream, data[i]);
     if (sz<data.getCommunicatedSize()) {
       T padding;
       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
	 mi1.marshall(stream, padding);
     }
     MarshallerInstance<unsigned> mi2;
     mi2.marshall(stream, data.size());
     if (data.size() != data.getCommunicatedSize()) {
       stream->requestRebuild(true);
       const_cast<ShallowArray<T>*>(&data)->setCommunicatedSize(data.size());
     }
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, ShallowArray<T> const & data) {
     MarshallerInstance<T> mi1;
     for (unsigned i=0; i < data.getCommunicatedSize(); ++i)
       mi1.getBlocks(blengths, blocs, data[i]);
     MarshallerInstance<unsigned> mi2;
     mi2.getBlocks(blengths, blocs, data.getCommunicatedSize());
   }
};

template <class T>
class MarshallerInstance<ShallowArray<T,3,2> > 
{
public:
  void marshall(OutputStream* stream, ShallowArray<T,3,2> const & data) {
     MarshallerInstance<T> mi1;
     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
     for (unsigned i=0; i < sz; ++i)
       mi1.marshall(stream, data[i]);
     if (sz<data.getCommunicatedSize()) {
       T padding;
       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
	 mi1.marshall(stream, padding);
     }
     MarshallerInstance<unsigned> mi2;
     mi2.marshall(stream, data.size());
     if (data.size() != data.getCommunicatedSize()) {
       stream->requestRebuild(true);
       const_cast<ShallowArray<T,3,2>*>(&data)->setCommunicatedSize(data.size());
     }
   }
  void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, ShallowArray<T,3,2> const & data) {
     MarshallerInstance<T> mi1;
     for (unsigned i=0; i < data.getCommunicatedSize(); ++i)
       mi1.getBlocks(blengths, blocs, data[i]);
     MarshallerInstance<unsigned> mi2;
     mi2.getBlocks(blengths, blocs, data.getCommunicatedSize());
   }
};

template <class T>
class MarshallerInstance<DeepPointerArray<T> >
{
public:
   void marshall(OutputStream* stream, DeepPointerArray<T> const & data) {
     MarshallerInstance<T> mi1;
     unsigned sz = (data.getCommunicatedSize() < data.size()) ? data.getCommunicatedSize() : data.size();
     for (unsigned i=0; i < sz; ++i)
       mi1.marshall(stream, data[i]);
     if (sz<data.getCommunicatedSize()) {
       T padding;
       for (unsigned i=sz; i < data.getCommunicatedSize(); ++i) 
	 mi1.marshall(stream, padding);
     }
     MarshallerInstance<unsigned> mi2;
     mi2.marshall(stream, data.size());
     if (data.size() != data.getCommunicatedSize()) {
       stream->requestRebuild(true);
       const_cast<DeepPointerArray<T>*>(&data)->setCommunicatedSize(data.size());
     }
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, DeepPointerArray<T> const & data) {
     MarshallerInstance<T> mi1;
     for (unsigned i=0; i < data.getCommunicatedSize(); ++i)
       mi1.getBlocks(blengths, blocs, data[i]);
     MarshallerInstance<unsigned> mi2;
     mi2.getBlocks(blengths, blocs, data.getCommunicatedSize());
   }
};

*/
#endif
#endif
