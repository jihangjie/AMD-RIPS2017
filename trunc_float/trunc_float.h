/* trunc_float.h
 * header file for truncated float class
 */

// allows int and float bitwise operations
typedef union {
  int as_int;
  float as_float;
} value_u;

class trunc_float {
  public:
    // constructors/deconstructor
    trunc_float();
    trunc_float(float value);
    trunc_float(float, int);
    ~trunc_float();
    // get and set methods
    int get_bitsize();
    float get_value();
    void set_bitsize(int);
    void set_value(float);
  private:
    int bitsize;
    float value;
};

