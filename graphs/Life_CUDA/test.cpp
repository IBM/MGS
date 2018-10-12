

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
