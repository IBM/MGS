// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CustomString_H
#define CustomString_H
#include "Copyright.h"

#include <iostream>

class CustomString
{
   public:
      // constructor
      CustomString();
      CustomString(const CustomString & str);
      CustomString(const char* cstr);
      CustomString(char fillCh, unsigned int count);

      // destructor
      ~CustomString();

      // value return methods
      unsigned int size();
      unsigned int capacity();

      // Function to return a blank string
      friend CustomString empty();

      // copy CustomString to c-string method
      void copy(char* cstr, unsigned int max);

      // create a c-string from CustomString method
      const char* c_str();

      // assignment method
      CustomString& operator = (const CustomString & str);

      // concatenation methods
      friend CustomString operator + (const CustomString & str1, const CustomString & str2);
      CustomString& operator += (const CustomString & str);

      // comparison methods
      int operator <  (const CustomString & str) const;
      int operator >  (const CustomString & str) const;
      int operator <= (const CustomString & str) const;
      int operator >= (const CustomString & str) const;
      int operator == (const CustomString & str) const;
      int operator != (const CustomString & str) const;

      int compare(const CustomString & str) const;

      // substring search methods
      int find(const CustomString & str, unsigned int & pos);

      // substring deletion method
      void del(unsigned int pos, unsigned int count);

      // substring insertion methods
      void insert(unsigned int pos, char ch);
      void insert(unsigned int pos, const CustomString & str);

      // append method
      void append(char ch);

      // substring retrieval method
      CustomString subStr(unsigned int start, unsigned int count);

      // character retrieval method
      char operator [] (unsigned int pos);

      // case-modification methods
      CustomString toUpper();
      CustomString toLower();

      // stream I/O methods
      friend std::istream & operator >> (std::istream & input, CustomString & str);
      friend std::ostream & operator << (std::ostream & output, const CustomString & str);

   private:
      enum StrCompVal  {SC_LESS = -1, SC_EQUAL = 0, SC_GREATER = 1};
      // instance variables
      unsigned int _capacity;    // allocated capacity
      unsigned int _size;       // current size
      char* _data = 0;             // pointer to text

      // class constant
      static unsigned int AllocIncr;

      // private method used to shrink a string to its minimum allocation
      void shrink();
      void reset();

};


#endif // CustomString_H
