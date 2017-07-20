// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      int s = data.size();
      *stream << s;
      *stream << data;
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, std::string const& data) {
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
     assert(0);
     // UNFINISHED: DemarshallerInstance.h and MarshallCommon.h must also be modified
     *stream << *data;
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, NDPairList* data) {
     assert(0);
     // UNFINISHED: DemarshallerInstance.h and MarshallCommon.h must also be modified
   }
};

template <class T>
class MarshallerInstance<Array<T> >
{
public:
   void marshall(OutputStream* stream, Array<T> const& data) {
      int s = data.size();
      *stream << s;
      MarshallerInstance<T> mi;
      for (int i=0; i < s; ++i)
         mi.marshall(stream, data[i]);
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, Array<T> const& data) {
     MarshallerInstance<T> mi;
     typename Array<T>::const_iterator iter, end=data.end();
     for (iter=data.begin(); iter!=end; ++iter)
       mi.getBlocks(blengths, blocs, *iter);
   }
};

template <class T>
class MarshallerInstance<ShallowArray<T> >
{
public:
   void marshall(OutputStream* stream, ShallowArray<T> const& data) {
      int s = data.size();
      *stream << s;
      MarshallerInstance<T> mi;
      for (int i=0; i < s; ++i)
         mi.marshall(stream, data[i]);
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, ShallowArray<T> const& data) {
     MarshallerInstance<T> mi;
     typename ShallowArray<T>::const_iterator iter, end=data.end();
     for (iter=data.begin(); iter!=end; ++iter)
       mi.getBlocks(blengths, blocs, *iter);
   }
};

template <class T>
class MarshallerInstance<ShallowArray<T,3,2> > 
{
public:
   void marshall(OutputStream* stream, ShallowArray<T,3,2> const& data) {
      int s = data.size();
      *stream << s;
      MarshallerInstance<T> mi;
      for (int i=0; i < s; ++i)
         mi.marshall(stream, data[i]);
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, ShallowArray<T,3,2> const& data) {
     MarshallerInstance<T> mi;
     typename ShallowArray<T,3,2>::const_iterator iter, end=data.end();
     for (iter=data.begin(); iter!=end; ++iter)
       mi.getBlocks(blengths, blocs, *iter);
   }
};

template <class T>
class MarshallerInstance<DeepPointerArray<T> >
{
public:
   void marshall(OutputStream* stream, DeepPointerArray<T> const& data) {
      int s = data.size();
      *stream << s;
      MarshallerInstance<T> mi;
      for (int i=0; i < s; ++i)
         mi.marshall(stream, data[i]);
   }
   void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs, DeepPointerArray<T> const& data) {
     MarshallerInstance<T> mi;
     typename DeepPointerArray<T>::const_iterator iter, end=data.end();
     for (iter=data.begin(); iter!=end; ++iter)
       mi.getBlocks(blengths, blocs, *iter);
   }
};

#endif
#endif
