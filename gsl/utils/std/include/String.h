// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef String_H
#define String_H
#include "Copyright.h"

#include <iostream>

class String
{
   public:
      // constructor
      String();
      String(const String & str);
      String(const char* cstr);
      String(char fillCh, unsigned int count);

      // destructor
      ~String();

      // value return methods
      unsigned int size();
      unsigned int capacity();

      // Function to return a blank string
      friend String empty();

      // copy String to c-string method
      void copy(char* cstr, unsigned int max);

      // create a c-string from String method
      const char* c_str();

      // assignment method
      String& operator = (const String & str);

      // concatenation methods
      friend String operator + (const String & str1, const String & str2);
      String& operator += (const String & str);

      // comparison methods
      int operator <  (const String & str) const;
      int operator >  (const String & str) const;
      int operator <= (const String & str) const;
      int operator >= (const String & str) const;
      int operator == (const String & str) const;
      int operator != (const String & str) const;

      int compare(const String & str) const;

      // substring search methods
      int find(const String & str, unsigned int & pos);

      // substring deletion method
      void del(unsigned int pos, unsigned int count);

      // substring insertion methods
      void insert(unsigned int pos, char ch);
      void insert(unsigned int pos, const String & str);

      // append method
      void append(char ch);

      // substring retrieval method
      String subStr(unsigned int start, unsigned int count);

      // character retrieval method
      char operator [] (unsigned int pos);

      // case-modification methods
      String toUpper();
      String toLower();

      // stream I/O methods
      friend std::istream & operator >> (std::istream & input, String & str);
      friend std::ostream & operator << (std::ostream & output, const String & str);

   private:
      enum StrCompVal  {SC_LESS = -1, SC_EQUAL = 0, SC_GREATER = 1};
      // instance variables
      unsigned int _capacity;    // allocated capacity
      unsigned int _size;       // current size
      char* _data;             // pointer to text

      // class constant
      static unsigned int AllocIncr;

      // private method used to shrink a string to its minimum allocation
      void shrink();

};


#endif // String_H
