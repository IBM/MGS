// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================



void main() {
   // create array of 5 integers
   {
      int * p1 = new int[5];
      delete[] p1;
   }
   {
      int * p1 = new int[1];
      delete[] p1;
   }
   {
      int * p1 = new int[5];
      delete[] (void*)p1;
   }
}
