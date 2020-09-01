/* src/config_coinutils.h.  Generated from config_coinutils.h.in by configure.  */
/* config_coinutils.h.in */

#ifndef __CONFIG_COINUTILS_H__
#define __CONFIG_COINUTILS_H__

/* Library Visibility Attribute */
#define COINUTILSLIB_EXPORT __declspec(dllimport)

/* Define to 1 if CoinUtils uses C++11 */
/* #undef COINUTILS_CPLUSPLUS11 */

/* Define to 1 if stdint.h is available for CoinUtils */
#define COINUTILS_HAS_STDINT_H 1

/* Define to 1 if stdint.h is available for CoinUtils */
#define COINUTILS_HAS_CSTDINT 1

/* Define to 1 if CoinUtils was build with Glpk support enabled */
#define COINUTILS_HAS_GLPK 1

/* Define to be the name of C-function for Inf check */
#define COINUTILS_C_FINITE std::isfinite

/* Define to be the name of C-function for NaN check */
#define COINUTILS_C_ISNAN std::isnan

/* Version number of project */
#define COINUTILS_VERSION "devel"

/* Major Version number of project */
#define COINUTILS_VERSION_MAJOR 9999

/* Minor Version number of project */
#define COINUTILS_VERSION_MINOR 9999

/* Release Version number of project */
#define COINUTILS_VERSION_RELEASE 9999

/* Define to 64bit integer type */
#define COINUTILS_INT64_T int64_t

/* Define to integer type capturing pointer */
#define COINUTILS_INTPTR_T intptr_t

/* Define to 64bit unsigned integer type */
#define COINUTILS_UINT64_T uint64_t

/* Define to type of CoinBigIndex */
#define COINUTILS_BIGINDEX_T int

/* Define to 1 if CoinBigIndex is int */
#define COINUTILS_BIGINDEX_IS_INT 1

#endif
