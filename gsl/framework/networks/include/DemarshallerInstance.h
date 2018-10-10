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

#ifndef DEMARSHALLERINSTANCE_H
#define DEMARSHALLERINSTANCE_H
#include "Copyright.h"

#include <mpi.h>
#include "Demarshaller.h"
#include "ShallowArray.h"
#include "DeepPointerArray.h"
#include "NDPairList.h"
#include "Simulation.h"
#include <memory>
#include <string.h>
#include <string>
#include <cassert>

template <class T> 
class DemarshallerInstance : public Demarshaller
{
public:
   DemarshallerInstance()
     : _destination(0), _offset(0)
   {
   }
   DemarshallerInstance(T* destination)
     : _offset(0)
   {
     _destination = reinterpret_cast<char*>(destination);
   }
   void setDestination(T *destination) {
     _destination = reinterpret_cast<char*>(destination);
     reset();
   }
   virtual void reset(){
      _offset  = 0;
   }
   virtual bool done() {
     return (_offset == sizeof(T));
   }
   virtual void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs)
   {
     blengths.push_back(sizeof(T));
     MPI_Aint blockAddress;
     MPI_Get_address(_destination, &blockAddress);
     blocs.push_back(blockAddress);
   } 
   virtual int demarshall(const char * buffer, int size, bool& rebuildRequested) // returns bytes remaining in the buffer
   {
       int retval = size;
       if (!done()) {
          int bytesRemaining = sizeof(T) - _offset;
          int toTransfer = (bytesRemaining<size)?bytesRemaining:size;
          //memcpy(_destination+_offset, buffer, toTransfer);
          //TUAN updated
	  std::copy(buffer, buffer+toTransfer, _destination+_offset);
	  _offset += toTransfer;
          retval = size - toTransfer;
       }
       return retval;
   }
   ~DemarshallerInstance() {}
   
private:
   char *_destination;
   int _offset;
};

template <class T> 
class DemarshallerInstance<Array<T> >: public Demarshaller {
public:
   DemarshallerInstance(Array<T> * destination)
     : _arrayDestination(destination), _arrayIndex(0), _arraySize(0), _arraySizeDemarshaller(&_arraySize)
   {
       _arrayElementDemarshaller = new DemarshallerInstance<T>(); // must set this up later
   }
   DemarshallerInstance()
     : _arrayDestination(0), _arrayIndex(0), _arraySize(0), _arraySizeDemarshaller(&_arraySize)
   {
       _arrayElementDemarshaller = new DemarshallerInstance<T>(); // must set this up later
   }
   void setDestination(Array<T> *arrayDestination) {
      _arrayDestination = arrayDestination;
      reset();
   }
   virtual void reset() {
      _arrayIndex = 0;
      _arraySize = _arrayDestination->size();
      _arraySizeDemarshaller.reset();
      _arrayElementDemarshaller->reset();
   }
   virtual bool done() {
     return ( (_arraySize==0 || 
	      ( _arrayIndex == _arraySize && 
		_arrayElementDemarshaller->done() ) ) 
	      && _arraySizeDemarshaller.done()
	      );
   }
  /* TODO TUAN check if we need to modify to work with flat array for GPU here */
   virtual void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs)
   {
     if (_arraySize < _arrayDestination->getSizeToCommunicate()) {
       _arraySize = _arrayDestination->getSizeToCommunicate();
       _arrayDestination->increaseSizeTo(_arraySize);
     }
     else if (_arraySize > _arrayDestination->getSizeToCommunicate()) {
       _arraySize = _arrayDestination->getSizeToCommunicate();
       _arrayDestination->decreaseSizeTo(_arraySize);
     }
     typename Array<T>::iterator iter, end=_arrayDestination->end();
     for (iter=_arrayDestination->begin(); iter!=end; ++iter) {
       _arrayElementDemarshaller->setDestination(&(*iter));	     
       _arrayElementDemarshaller->getBlocks(blengths, blocs);
     }
     _arraySizeDemarshaller.getBlocks(blengths, blocs);
   } 
   virtual int demarshall(const char * buffer, int size, bool& rebuildRequested) // returns bytes remaining in the buffer
   {
     const char* buff = buffer;
     int buffSize = size;
     assert(buffSize>0);
     while ( _arraySize>0 &&
	     (_arrayIndex<_arraySize || !_arrayElementDemarshaller->done() ) ) {
       _arrayElementDemarshaller->setDestination(&(*_arrayDestination)[_arrayIndex]);
       buffSize = _arrayElementDemarshaller->demarshall(buff, buffSize, rebuildRequested);       
       assert(buffSize>0);
       buff = buffer+(size-buffSize);
       if (_arrayElementDemarshaller->done()) ++_arrayIndex;
     }
     while (!_arraySizeDemarshaller.done()) {
       buffSize = _arraySizeDemarshaller.demarshall(buff, buffSize, rebuildRequested);
       buff = buffer+(size-buffSize);
     }
     if (_arraySize != _arrayDestination->size()) {
       rebuildRequested = true;
       _arrayDestination->setSizeToCommunicate(_arraySize);
       _arraySize=_arrayDestination->size();
     }
     return buffSize;
   }
   ~DemarshallerInstance()
   {
       delete _arrayElementDemarshaller;
   }

private:
   Array<T>* _arrayDestination;
   int _arrayIndex;
   unsigned _arraySize;

   DemarshallerInstance<unsigned> _arraySizeDemarshaller;
   DemarshallerInstance<T>* _arrayElementDemarshaller;
};

template <> class DemarshallerInstance<std::string>: public Demarshaller {
public:
   DemarshallerInstance(std::string * destination)
     : _stringDestination(destination), _stringIndex(0), _stringSize(0), _stringSizeDemarshaller(&_stringSize)
   {
   }
   DemarshallerInstance()
     : _stringDestination(0), _stringIndex(0), _stringSize(0), _stringSizeDemarshaller(&_stringSize)
   {
   }
   void setDestination(std::string *stringDestination) {
      _stringDestination = stringDestination;
      reset();
   }
   virtual void reset(){
      _stringIndex = _stringSize = 0;
      _stringSizeDemarshaller.reset();
      *_stringDestination = "";
   }
   virtual bool done() {
       return ( _stringSizeDemarshaller.done() &&
                (_stringIndex == _stringSize) );
   }
   virtual void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs)
   {
       blengths.push_back(_stringDestination->size());
       MPI_Aint blockAddress;
       MPI_Get_address(const_cast<char*>(_stringDestination->c_str()), &blockAddress);
       blocs.push_back(blockAddress);
   } 
   virtual int demarshall(const char * buffer, int size, bool& rebuildRequested) // returns bytes remaining in the buffer
   {
       const char* buff = buffer;
       int buffSize = size;
       if (!_stringSizeDemarshaller.done()) {
         buffSize = _stringSizeDemarshaller.demarshall(buffer, size, rebuildRequested);
         buff = buffer+(size-buffSize);
         if (_stringSizeDemarshaller.done()) { 
	   *_stringDestination="";
         }
       }
       if (!done() && buffSize!=0) {
	 int bytesRemaining = _stringSize - _stringIndex;
	 int toTransfer = (bytesRemaining<buffSize)?bytesRemaining:buffSize;
	 _stringDestination->append(buff, toTransfer);
	 buffSize -= toTransfer;
	 _stringIndex += toTransfer;
       }
       return buffSize;
   }
   ~DemarshallerInstance()
   {
   }

private:
   std::string* _stringDestination;
   int _stringIndex;
   int _stringSize;

   DemarshallerInstance<int> _stringSizeDemarshaller;
};

template <class T> class DemarshallerInstance<ShallowArray<T> > : public DemarshallerInstance<Array<T> >
{
public:
   DemarshallerInstance(ShallowArray<T> * destination)
     : DemarshallerInstance<Array<T> >(destination)
   {
   }
   DemarshallerInstance()
     : DemarshallerInstance<Array<T> >()
   {
   }
   void setDestination(ShallowArray<T> *arrayDestination) 
   {
      DemarshallerInstance<Array<T> >::setDestination(arrayDestination);
   }
};
template <class T> class DemarshallerInstance<ShallowArray<T,3,2> > : public DemarshallerInstance<Array<T> >
{
public:
   DemarshallerInstance(ShallowArray<T,3,2> * destination)
     : DemarshallerInstance<Array<T> >(destination)
   {
   }
   DemarshallerInstance()
     : DemarshallerInstance<Array<T> >()
   {
   }
   void setDestination(ShallowArray<T,3,2> *arrayDestination) 
   {
      DemarshallerInstance<Array<T> >::setDestination(arrayDestination);
   }
};

template <class T> class DemarshallerInstance<DeepPointerArray<T> > : public DemarshallerInstance<Array<T> >
{
public:
   DemarshallerInstance(DeepPointerArray<T> * destination)
     : DemarshallerInstance<Array<T> >(destination)
   {
   }
   DemarshallerInstance()
     : DemarshallerInstance<Array<T> >()
   {
   }
   void setDestination(DeepPointerArray<T> *arrayDestination) 
   {
      DemarshallerInstance<Array<T> >::setDestination(arrayDestination);
   }
};


/*

UNIFINISHED: Marshall.h and MarshallCommon.h must also be modified. Consider making NDPairlist use a ShallowArray of NDPairs - JK

template <> class DemarshallerInstance<NDPairList>: public Demarshaller {
public:
   DemarshallerInstance(NDPairList * destination)
     : _ndplDestination(destination), _ndplIndex(0), _ndplSize(0), _ndplSizeDemarshaller(&_ndplSize)
   {
   }
   DemarshallerInstance()
     : _ndplDestination(0), _ndplIndex(0), _ndplSize(0), _ndplSizeDemarshaller(&_ndplSize)
   {
   }
   void setDestination(NDPairList *ndplDestination) {
      _ndplDestination = ndplDestination;
      reset();
   }
   virtual void reset() {
      _ndplIndex = 0;
      _ndplSizeDemarshaller.reset();
      _ndplElementDemarshaller.reset();
   }
   virtual bool done() {
       if (_ndplSize==0 && _ndplSizeDemarshaller.done()) assert(_ndplElementDemarshaller->done());
       return ( _ndplSizeDemarshaller.done() && 
		(_ndplIndex == _ndplSize) && 
		_ndplElementDemarshaller->done() );

   }
   virtual int demarshall(const char * buffer, int size) // returns bytes remaining in the buffer
   {
       const char* buff = buffer;
       int buffSize = size;
       if (!_ndplSizeDemarshaller.done()) {
	 buffSize = _ndplSizeDemarshaller.demarshall(buffer, size);
	 buff = buffer+(size-buffSize);
	 if (_ndplSizeDemarshaller.done()) {
           if (_ndplSize != _ndplDestination->size()){
	      if (_ndplSize < _ndplDestination->size()) _ndplDestination->clear();
	      _ndplDestination->increaseSizeTo(_ndplSize);	
           }
	   _ndplElementDemarshaller->setDestination(&(*_ndplDestination)[0]);
	 }
       }
       if (!done() && buffSize!=0) {
	 while (_ndplIndex<_ndplSize && buffSize>0) {
	   buffSize = _ndplElementDemarshaller->demarshall(buff, buffSize);
	   buff = buffer+(size-buffSize);
	   if (_ndplElementDemarshaller->done()) {
	     ++_ndplIndex;
	     if (_ndplIndex<_ndplSize)
	       _ndplElementDemarshaller->setDestination(&(*_ndplDestination)[_ndplIndex]);	     
	   }
	 }
       }
       return buffSize;
   }
   ~DemarshallerInstance()
   {
       delete _ndplElementDemarshaller;
   }

private:
   NDPairList* _ndplDestination;
   int _ndplIndex;
   int _ndplSize;

   DemarshallerInstance<int> _ndplSizeDemarshaller;
   DemarshallerInstance<> _ndplElementDemarshaller;
};

*/

#endif
