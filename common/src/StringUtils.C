#include "../include/StringUtils.h"
#include <string>
#include <limits>
#include <iostream>

using namespace std;

// GOAL: parse the given string
//  into a vector of tokens
//  using the given set of delimiters
// DEFAULT: delimiter is blank space
void StringUtils::Tokenize(const string& str,
                           vector<string>& tokens,
                           const string& delimiters)

{/* It uses any character in the string 'delimiters'-argument as the delimiter
	*/
    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    string::size_type pos = str.find_first_of(delimiters, lastPos);

    tokens.clear(); /* make sure the vector is empty */
    while (string::npos != pos || string::npos != lastPos) {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

// GOAL: just print out a message - pause the execution 
// until user presses ENTER key
void StringUtils::wait()
{
  std::cout << "Press ENTER to continue...";
  std::cin.ignore( std::numeric_limits < std::streamsize> :: max(), '\n');
}
