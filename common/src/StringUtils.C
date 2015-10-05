#include "../include/StringUtils.h"
#include <string>

using namespace std;

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
