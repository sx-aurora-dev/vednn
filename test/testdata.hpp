#ifndef TESTDATA_HPP
#define TESTDATA_HPP

#include "conv_test_param.h"
#include "vednn-def.h"
#include <cstdint>
#include <vector>

#define MAX_TEST_NAME 100

// TODO make a C++ object to avoid code duplication for TestData
/** New stats, by {test,impl} */
struct TestData {
    unsigned long long sum_times;
    size_t test;
    size_t reps;
    double diff;
    uint64_t ops;
    int impl_idx;
    int impl_type; // 0/1/2 for libvednn-std/lbivednn-impl/JIT-impl
    char test_name[MAX_TEST_NAME];
    char impl_name[MAX_TEST_NAME]; ///< shorter
    char descr[MAX_TEST_NAME];     ///< longer (only for impl_type 2=JIT)
    // TODO: size_t dup_of; ///< if removed due to duplication, keep that info
    TestData( int const test, char const* test_name,
            int const impl_idx, char const* impl_name, int impl_type,
            char const* descr=nullptr );
    TestData()
        : sum_times(0ULL), test(0U), reps(0U), diff(-13.0), ops(0ULL),
        impl_idx(-1), impl_type(-1), test_name("test_name"),
        impl_name("impl_name"), descr("")
    {}
    void setDescr(char const* description); ///< parm string, comment
    void appendDescr(char const* description); ///< parm string, comment
};

struct TestDataRepo : public std::vector<TestData>
{
    double const hertz;
    TestDataRepo(double hertz) : std::vector<TestData>(), hertz(hertz){}

    /** add some new data, print incoming [default]. */
    void append(std::vector<TestData> const& vtd, int const verbose=1);
    void append(TestData const& td, int const verbose=1);
    void print() const;
    void wins() const;
};

/** summarize a single test \c t (an implementation function) fastest-first,
 * selected from \c test_data[a..z].*/
void print_test_data_single( struct TestData const* test_data, int const t,
        int const a, int const z,
        double const HZ, char const* header=nullptr );

/** print td[a] to td[z] sort by test (todo: then increasing avg time).
 * \c v verbosity[1] (if 0, caller might want to output newline). */
void print_test_data( struct TestData const* test_data, int const a, int const z,
        double const HZ, int const v=1);

/** print \b all (nEntry) tests (impls) in \c test_data with nice test label line */
void print_test_data( struct TestDataRepo const& tdRepo,
        struct param const *pNetwork, int nEntry );

// vim: et ts=4 sw=4 cindent cino=^0,=0,l0,\:0,N-s syntax=cpp.doxygen
#endif //TESTDATA_HPP
